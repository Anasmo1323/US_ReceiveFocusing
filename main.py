import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QGroupBox, QTabWidget)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- PHYSICS CONSTANTS ---
C = 1540.


class UltrasoundPhysicsEngine:
    def __init__(self):
        self.num_elements = 64
        self.pitch = 0.3e-3
        width = (self.num_elements - 1) * self.pitch
        self.xi = np.linspace(-width / 2, width / 2, self.num_elements)
        self.fc = 5.0e6
        self.active_elements = 64

        # Grid for Tab 1 (Beam Map)
        self.nz = 200
        self.nx = 150
        self.z_max = 100e-3
        self.x_max = 20e-3
        z_axis = np.linspace(1e-3, self.z_max, self.nz)
        x_axis = np.linspace(-self.x_max, self.x_max, self.nx)
        self.X_grid, self.Z_grid = np.meshgrid(x_axis, z_axis)
        
        # B-mode imaging parameters
        self.n_lines = 80  # Number of scan lines
        self.n_samples = 150  # Depth samples per line
        self.scan_range = 15e-3  # Lateral scan range (±15mm)
        self.depth_range = 100e-3  # Max depth
        
        # Generate distributed scatterer field
        self.generate_scatterer_field()

    def calculate_single_beam_field(self, focal_depth):
        """ Calculates the 2D Beam field for Tab 1 """
        k = 2 * np.pi * self.fc / C
        dist_focus_to_element = np.sqrt(self.xi ** 2 + focal_depth ** 2)
        dist_focus_center = focal_depth

        # Focusing Phase (The Electronic Lens)
        # Phase shift to compensate for path difference
        path_diff = dist_focus_to_element - dist_focus_center
        focusing_phase = -k * path_diff

        # Grid Calculations
        DX = self.X_grid[..., np.newaxis] - self.xi[np.newaxis, np.newaxis, :]
        DZ = self.Z_grid[..., np.newaxis]
        dist_grid = np.sqrt(DX ** 2 + DZ ** 2)

        total_phase = k * dist_grid + focusing_phase[np.newaxis, np.newaxis, :]
        complex_sum = np.sum(np.exp(1j * total_phase), axis=2)

        mag = np.abs(complex_sum)
        mag = mag / (np.max(mag) + 1e-12)
        img_db = 20 * np.log10(mag + 1e-12)

        # Calculate Delays for visualization
        delays = path_diff / C

        return np.clip(img_db, -60, 0), delays

    def simulate_b_mode_resolution(self, fixed_depth):
        """
        Simulates the Lateral Resolution (Beam Width) for Case A vs Case B.
        Instead of full wave sim (slow), we calculate the theoretical beam width
        at specific target depths.
        """
        targets = np.array([20, 40, 60, 80]) * 1e-3  # Target depths

        res_A = []  # Resolution for Case A (Fixed)
        res_B = []  # Resolution for Case B (Dynamic)

        D = self.active_elements * self.pitch
        lam = C / self.fc

        for z_target in targets:
            # --- CASE B: Dynamic Focus ---
            # Focus is perfectly at z_target.
            # Width is diffraction limited spot size: F# * lambda
            # F# = z_target / D
            # Spot = z_target * lambda / D
            width_dynamic = (z_target * lam) / D
            res_B.append(width_dynamic)

            # --- CASE A: Fixed Focus ---
            # Focus is at fixed_depth (Zf). Target is at z_target (z).
            # Beam width is Spot Size at Zf + Geometric Defocus

            # 1. Width at the fixed focus
            w0 = (fixed_depth * lam) / D

            # 2. Defocusing spread
            # Geometry: The beam converges to w0 at Zf, then spreads.
            # Spread angle approx D / Zf (actually determined by aperture)
            distance_off_focus = abs(z_target - fixed_depth)

            # Simple geometric model of defocusing
            # At aperture (z=0), width is D. At focus (z=Zf), width is w0.
            # Slope m = (D - w0) / Zf
            if z_target < fixed_depth:
                # Converging region
                width_fixed = w0 + (distance_off_focus / fixed_depth) * (D - w0)
            else:
                # Diverging region
                # Divergence angle theta. tan(theta) approx D / (2*Zf) or diffraction
                divergence_factor = D / fixed_depth
                width_fixed = w0 + distance_off_focus * divergence_factor

            res_A.append(width_fixed)

        return targets, np.array(res_A), np.array(res_B)
    
    def measure_bmode_resolution(self, x_lines, z_samples, bmode_image):
        """Measure lateral resolution from actual B-mode image data"""
        # Define depth positions to measure resolution
        measurement_depths = np.array([20, 40, 60, 80]) * 1e-3  # Same as before for comparison
        measured_widths = []
        
        for target_depth in measurement_depths:
            # Find closest depth index in B-mode image
            depth_idx = np.argmin(np.abs(z_samples - target_depth))
            
            # Extract lateral profile at this depth
            lateral_profile = bmode_image[depth_idx, :]
            
            # Find peak position (center of beam)
            peak_idx = np.argmax(lateral_profile)
            peak_value = lateral_profile[peak_idx]
            
            # Calculate -6dB threshold (FWHM measurement standard)
            threshold_6db = peak_value - 6.0  # 6dB below peak
            
            # Find left and right edges at -6dB threshold
            left_edge = peak_idx
            right_edge = peak_idx
            
            # Search left from peak
            for i in range(peak_idx, -1, -1):
                if lateral_profile[i] < threshold_6db:
                    left_edge = i
                    break
            
            # Search right from peak  
            for i in range(peak_idx, len(lateral_profile)):
                if lateral_profile[i] < threshold_6db:
                    right_edge = i
                    break
            
            # Convert indices to actual lateral distance
            if right_edge > left_edge:
                width_indices = right_edge - left_edge
                lateral_spacing = (x_lines[-1] - x_lines[0]) / (len(x_lines) - 1)
                measured_width = width_indices * lateral_spacing
            else:
                # Fallback: use minimum measurable width
                lateral_spacing = (x_lines[-1] - x_lines[0]) / (len(x_lines) - 1)
                measured_width = 2 * lateral_spacing  # 2-pixel minimum
            
            measured_widths.append(measured_width)
        
        return measurement_depths, np.array(measured_widths)
    
    def calculate_depth_of_field(self, focus_depth, strategy='fixed'):
        """Calculate depth of field - axial range with acceptable focus quality"""
        # DOF definition: range where beam width ≤ √2 × minimum width (3dB criterion)
        # Or alternatively: range where intensity ≥ 50% of peak (6dB criterion)
        
        D = self.active_elements * self.pitch
        lam = C / self.fc
        
        if strategy == 'dynamic':
            # Dynamic focusing: DOF is theoretically infinite since focus tracks depth
            # In practice, limited by system constraints, but much larger than fixed
            dof_range = self.depth_range - 10e-3  # Nearly full imaging range
            dof_start = 10e-3
            dof_end = self.depth_range
            return dof_range, dof_start, dof_end
        
        else:  # Fixed focusing
            # Calculate DOF around fixed focus depth
            # Minimum beam width at focus
            w_min = (focus_depth * lam) / D
            
            # DOF criterion: width ≤ √2 × w_min
            w_threshold = w_min * np.sqrt(2)
            
            # Find depth range where beam width meets criterion
            # Search range around focus
            search_depths = np.linspace(5e-3, self.depth_range, 200)
            
            dof_start = focus_depth
            dof_end = focus_depth
            
            # Find DOF boundaries
            for z in search_depths:
                beam_width = self.calculate_beam_width(z, focus_depth)
                if beam_width <= w_threshold:
                    dof_start = min(dof_start, z)
                    dof_end = max(dof_end, z)
            
            dof_range = dof_end - dof_start
            return dof_range, dof_start, dof_end
    
    def generate_scatterer_field(self):
        """Generate a field of randomly distributed scatterers"""
        np.random.seed(42)  # For reproducible results
        
        # Create a more realistic scatterer density that the US system can resolve
        # Typical resolution: ~0.5mm lateral x 0.3mm axial
        # So we need scatterers spaced at least this far apart to be resolvable
        n_scatterers = 400  # Reduced from 2000 for realistic resolution
        
        # Random positions within imaging region
        self.scat_x = np.random.uniform(-self.scan_range, self.scan_range, n_scatterers)
        self.scat_z = np.random.uniform(10e-3, self.depth_range, n_scatterers)
        
        # Random scattering amplitudes (reflectivity)
        self.scat_amp = np.random.normal(1.0, 0.3, n_scatterers)
        self.scat_amp = np.abs(self.scat_amp)  # Positive amplitudes
    
    def simulate_bmode_line(self, beam_x, focus_strategy, focus_depth=None):
        """Simulate B-mode acquisition for a single scan line"""
        k = 2 * np.pi * self.fc / C
        
        # Define depth samples
        z_samples = np.linspace(5e-3, self.depth_range, self.n_samples)
        line_intensity = np.zeros(self.n_samples)
        
        for i, z_sample in enumerate(z_samples):
            # Determine receive focus depth
            if focus_strategy == 'fixed':
                focus_z = focus_depth
            else:  # dynamic
                focus_z = z_sample
            
            # Calculate receive focusing delays for this depth
            dist_focus = np.sqrt(self.xi**2 + focus_z**2)
            focusing_phase = -k * (dist_focus - focus_z)
            
            # Find scatterers contributing to this sample (within resolution cell)
            # Resolution cell size depends on focus quality
            cell_width = self.calculate_beam_width(z_sample, focus_z)
            cell_height = C / (2 * self.fc)  # Axial resolution
            
            # Select scatterers within this resolution cell
            x_mask = np.abs(self.scat_x - beam_x) < cell_width/2
            z_mask = np.abs(self.scat_z - z_sample) < cell_height/2
            active_mask = x_mask & z_mask
            
            if np.any(active_mask):
                # Calculate echo from active scatterers
                scat_x_active = self.scat_x[active_mask]
                scat_z_active = self.scat_z[active_mask]
                scat_amp_active = self.scat_amp[active_mask]
                
                # Distance from each element to each scatterer
                dx = scat_x_active[np.newaxis, :] - self.xi[:, np.newaxis]
                dz = scat_z_active[np.newaxis, :]
                distances = np.sqrt(dx**2 + dz**2)
                
                # Receive signal from each scatterer
                receive_phases = k * distances + focusing_phase[:, np.newaxis]
                receive_signals = scat_amp_active[np.newaxis, :] * np.exp(1j * receive_phases)
                
                # Sum across elements (coherent receive beamforming)
                coherent_sum = np.sum(receive_signals, axis=0)
                
                # Sum across scatterers (incoherent scattering)
                total_amplitude = np.abs(np.sum(coherent_sum))
                
                # Apply basic TGC (time gain compensation)
                tgc_factor = z_sample * 1000  # Simple depth-dependent gain
                line_intensity[i] = total_amplitude * tgc_factor
        
        return z_samples, line_intensity
    
    def calculate_beam_width(self, target_depth, focus_depth):
        """Calculate beam width at target depth given focus depth"""
        D = self.active_elements * self.pitch
        lam = C / self.fc
        
        if abs(target_depth - focus_depth) < 1e-6:  # At focus
            return (focus_depth * lam) / D
        
        # Off-focus beam width (simplified model)
        w0 = (focus_depth * lam) / D  # Width at focus
        defocus = abs(target_depth - focus_depth)
        
        if target_depth < focus_depth:
            # Converging region
            width = w0 + (defocus / focus_depth) * (D - w0)
        else:
            # Diverging region
            divergence = D / focus_depth
            width = w0 + defocus * divergence
        
        return width
    
    def generate_bmode_image(self, focus_strategy, focus_depth=None):
        """Generate full B-mode image using line-by-line processing"""
        # Define scan line positions
        x_lines = np.linspace(-self.scan_range, self.scan_range, self.n_lines)
        
        # Initialize image matrix
        bmode_image = np.zeros((self.n_samples, self.n_lines))
        
        # Process each scan line
        for line_idx, x_beam in enumerate(x_lines):
            z_samples, line_data = self.simulate_bmode_line(x_beam, focus_strategy, focus_depth)
            bmode_image[:, line_idx] = line_data
        
        # Convert to dB and apply dynamic range
        bmode_image = bmode_image / (np.max(bmode_image) + 1e-12)
        bmode_db = 20 * np.log10(bmode_image + 1e-6)
        bmode_db = np.clip(bmode_db, -60, 0)
        
        return x_lines, z_samples, bmode_db
    
    def generate_ground_truth_image(self):
        """Generate ground truth image showing actual scatterer positions"""
        # Use same grid resolution as B-mode for fair comparison
        x_gt = np.linspace(-self.scan_range, self.scan_range, self.n_lines)
        z_gt = np.linspace(5e-3, self.depth_range, self.n_samples)
        X_gt, Z_gt = np.meshgrid(x_gt, z_gt)
        
        # Initialize ground truth image
        gt_image = np.zeros_like(X_gt)
        
        # Add each scatterer with realistic point spread function
        # Use beam width as the PSF size for fair comparison
        for i in range(len(self.scat_x)):
            # Find closest grid points
            x_idx = np.argmin(np.abs(x_gt - self.scat_x[i]))
            z_idx = np.argmin(np.abs(z_gt - self.scat_z[i]))
            
            # Calculate realistic PSF size at this depth
            beam_width = self.calculate_beam_width(self.scat_z[i], self.scat_z[i])  # Ideal focus
            sigma_x = beam_width / 4  # Lateral PSF
            sigma_z = C / (4 * self.fc)  # Axial PSF (pulse length)
            
            # Add Gaussian blob representing this scatterer
            for dx in range(-2, 3):  # 5x5 kernel
                for dz in range(-2, 3):
                    xi = x_idx + dx
                    zi = z_idx + dz
                    if 0 <= xi < len(x_gt) and 0 <= zi < len(z_gt):
                        dist_x = (x_gt[xi] - self.scat_x[i])**2 / (2 * sigma_x**2)
                        dist_z = (z_gt[zi] - self.scat_z[i])**2 / (2 * sigma_z**2)
                        gt_image[zi, xi] += self.scat_amp[i] * np.exp(-(dist_x + dist_z))
        
        # Normalize and convert to dB (same processing as B-mode)
        gt_image = gt_image / (np.max(gt_image) + 1e-12)
        gt_db = 20 * np.log10(gt_image + 1e-6)
        gt_db = np.clip(gt_db, -60, 0)
        
        return x_gt, z_gt, gt_db


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Task 8: Ultrasound Simulation Suite")
        self.resize(1200, 900)

        self.engine = UltrasoundPhysicsEngine()

        # Main Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- TAB 1: PHYSICS LAB ---
        self.tab1 = QWidget()
        self.setup_tab1()
        self.tabs.addTab(self.tab1, "Tab 1: Physics Lab (Single Beam)")

        # --- TAB 2: B-MODE IMAGING ---
        self.tab2 = QWidget()
        self.setup_tab2()
        self.tabs.addTab(self.tab2, "Tab 2: Case A vs Case B (B-Mode)")

    def setup_tab1(self):
        layout = QHBoxLayout(self.tab1)

        # Controls
        controls = QVBoxLayout()
        grp = QGroupBox("Electronic Lens")
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("<b>Fixed Focus Depth (mm)</b>"))
        self.slider_focus_t1 = QSlider(Qt.Horizontal)
        self.slider_focus_t1.setRange(10, 90)
        self.slider_focus_t1.setValue(40)
        self.slider_focus_t1.valueChanged.connect(self.update_tab1)
        vbox.addWidget(self.slider_focus_t1)
        self.lbl_focus_t1 = QLabel("40 mm")
        vbox.addWidget(self.lbl_focus_t1)
        grp.setLayout(vbox)
        controls.addWidget(grp)
        controls.addStretch()
        layout.addLayout(controls, 1)

        # Plots
        self.fig1 = Figure()
        self.canvas1 = FigureCanvas(self.fig1)
        layout.addWidget(self.canvas1, 3)

        self.update_tab1()

    def update_tab1(self):
        depth = self.slider_focus_t1.value() * 1e-3
        self.lbl_focus_t1.setText(f"{depth * 1000:.0f} mm")

        img_db, delays = self.engine.calculate_single_beam_field(depth)

        self.fig1.clear()
        ax1 = self.fig1.add_subplot(121)
        ax1.imshow(img_db, extent=[-20, 20, 100, 0], cmap='jet', aspect='auto', vmin=-60, vmax=0)
        ax1.set_title("Beam Profile (Fixed Focus)")
        ax1.plot(0, depth * 1000, 'rx', markersize=10)

        ax2 = self.fig1.add_subplot(122)
        ax2.plot(self.engine.xi * 1000, delays * 1e6, 'c')
        ax2.set_title("Delay Curve")
        ax2.set_ylim(0, 1.5)
        ax2.grid(True)

        self.canvas1.draw()

    def setup_tab2(self):
        layout = QHBoxLayout(self.tab2)

        # Controls
        controls = QVBoxLayout()

        info = QLabel(
            "<b>Complete B-Mode Imaging Analysis:</b><br><br>"
            "<b>Ground Truth (Left):</b><br>"
            "Actual scatterer positions.<br>"
            "This is what we're trying to image.<br><br>"
            "<b>Case A (Fixed Focus):</b><br>"
            "Single receive focus depth.<br>"
            "Limited Depth of Field (DOF).<br>"
            "Green zone shows acceptable focus range.<br><br>"
            "<b>Case B (Dynamic Focus):</b><br>"
            "Focus tracks with depth.<br>"
            "Extended DOF across full range.<br><br>"
            "<b>Right:</b> MEASURED resolution + DOF<br>"
            "from actual B-mode images (-6dB width).<br>"
            "Solid = measured, dashed = theory.<br><br>"
            "<b>Try different frequencies and apertures</b><br>"
            "to see how system parameters affect<br>"
            "both resolution AND depth of field!"
        )
        info.setWordWrap(True)
        controls.addWidget(info)

        grp = QGroupBox("Case A Settings")
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Fixed Focus Depth:"))
        self.slider_focus_t2 = QSlider(Qt.Horizontal)
        self.slider_focus_t2.setRange(10, 90)
        self.slider_focus_t2.setValue(40)
        self.slider_focus_t2.valueChanged.connect(self.update_tab2)
        vbox.addWidget(self.slider_focus_t2)
        self.lbl_focus_t2 = QLabel("40 mm")
        vbox.addWidget(self.lbl_focus_t2)
        grp.setLayout(vbox)
        controls.addWidget(grp)
        
        # System Parameters
        grp2 = QGroupBox("System Parameters")
        vbox2 = QVBoxLayout()
        
        # Frequency control
        vbox2.addWidget(QLabel("Frequency (MHz):"))
        self.slider_freq = QSlider(Qt.Horizontal)
        self.slider_freq.setRange(25, 100)  # 2.5 to 10 MHz
        self.slider_freq.setValue(50)  # 5 MHz default
        self.slider_freq.valueChanged.connect(self.update_tab2)
        vbox2.addWidget(self.slider_freq)
        self.lbl_freq = QLabel("5.0 MHz")
        vbox2.addWidget(self.lbl_freq)
        
        # Aperture control  
        vbox2.addWidget(QLabel("Active Elements:"))
        self.slider_aperture = QSlider(Qt.Horizontal)
        self.slider_aperture.setRange(16, 64)  # 16 to 64 elements
        self.slider_aperture.setValue(64)  # Full aperture default
        self.slider_aperture.valueChanged.connect(self.update_tab2)
        vbox2.addWidget(self.slider_aperture)
        self.lbl_aperture = QLabel("64 elements")
        vbox2.addWidget(self.lbl_aperture)
        
        grp2.setLayout(vbox2)
        controls.addWidget(grp2)

        controls.addStretch()
        layout.addLayout(controls, 1)

        # Plots
        self.fig2 = Figure(figsize=(10, 8))
        self.canvas2 = FigureCanvas(self.fig2)
        layout.addWidget(self.canvas2, 3)

        self.update_tab2()

    def update_tab2(self):
        fixed_depth = self.slider_focus_t2.value() * 1e-3
        frequency = self.slider_freq.value() / 10.0 * 1e6  # Convert to Hz
        active_elements = self.slider_aperture.value()
        
        # Update labels
        self.lbl_focus_t2.setText(f"{fixed_depth * 1000:.0f} mm")
        self.lbl_freq.setText(f"{frequency/1e6:.1f} MHz")
        self.lbl_aperture.setText(f"{active_elements} elements")
        
        # Update engine parameters
        self.engine.fc = frequency
        self.engine.active_elements = active_elements
        
        # Regenerate scatterer field with new parameters (optional - for consistency)
        # self.engine.generate_scatterer_field()  # Uncomment if you want new random field

        self.fig2.clear()
        
        # Generate ground truth and B-mode images
        print(f"Generating complete imaging comparison... (Fixed focus at {fixed_depth*1000:.0f}mm)")
        
        # Ground Truth
        x_gt, z_gt, gt_image = self.engine.generate_ground_truth_image()
        
        # Case A: Fixed Focus
        x_lines_A, z_samples_A, bmode_A = self.engine.generate_bmode_image('fixed', fixed_depth)
        
        # Case B: Dynamic Focus  
        x_lines_B, z_samples_B, bmode_B = self.engine.generate_bmode_image('dynamic')
        
        # Create 4-panel layout: Ground Truth | Case A | Case B | Resolution Plot
        
        # Panel 1: Ground Truth
        ax1 = self.fig2.add_subplot(141)
        extent_gt = [x_gt[0]*1000, x_gt[-1]*1000, z_gt[-1]*1000, z_gt[0]*1000]
        ax1.imshow(gt_image, extent=extent_gt, cmap='hot', aspect='auto', vmin=-60, vmax=0)
        ax1.set_title('Ground Truth\n(Actual Scatterers)', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Depth (mm)')
        ax1.set_xlabel('Lateral (mm)')
        
        # Panel 2: Case A - Fixed Focus
        ax2 = self.fig2.add_subplot(142)
        extent_A = [x_lines_A[0]*1000, x_lines_A[-1]*1000, z_samples_A[-1]*1000, z_samples_A[0]*1000]
        ax2.imshow(bmode_A, extent=extent_A, cmap='gray', aspect='auto', vmin=-60, vmax=0)
        ax2.set_title('Case A: Fixed Focus\n(Single Focus Depth)', fontsize=9, fontweight='bold')
        ax2.set_xlabel('Lateral (mm)')
        ax2.set_yticklabels([])  # Remove y-axis labels
        
        # Mark the fixed focus depth
        ax2.axhline(fixed_depth * 1000, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.text(x_lines_A[0]*1000 + 1, fixed_depth * 1000 - 4, 'Focus', color='red', fontsize=7, fontweight='bold')
        
        # Panel 3: Case B - Dynamic Focus
        ax3 = self.fig2.add_subplot(143)
        extent_B = [x_lines_B[0]*1000, x_lines_B[-1]*1000, z_samples_B[-1]*1000, z_samples_B[0]*1000]
        ax3.imshow(bmode_B, extent=extent_B, cmap='gray', aspect='auto', vmin=-60, vmax=0)
        ax3.set_title('Case B: Dynamic Focus\n(Focus Tracks Depth)', fontsize=9, fontweight='bold')
        ax3.set_xlabel('Lateral (mm)')
        ax3.set_yticklabels([])  # Remove y-axis labels

        
        # Panel 4: Resolution Analysis - NOW USING ACTUAL IMAGE DATA
        print("Measuring resolution from actual B-mode images...")
        
        # Measure resolution from actual B-mode images
        z_targets_A, width_A_measured = self.engine.measure_bmode_resolution(x_lines_A, z_samples_A, bmode_A)
        z_targets_B, width_B_measured = self.engine.measure_bmode_resolution(x_lines_B, z_samples_B, bmode_B)
        
        ax4 = self.fig2.add_subplot(144)
        z_mm = z_targets_A * 1000  # Convert to mm
        w_A_mm = width_A_measured * 1000  # Convert to mm
        w_B_mm = width_B_measured * 1000  # Convert to mm

        ax4.plot(w_A_mm, z_mm, 'r-o', label='Case A: Fixed\n(Measured)', linewidth=2, markersize=4)
        ax4.plot(w_B_mm, z_mm, 'b-o', label='Case B: Dynamic\n(Measured)', linewidth=2, markersize=4)
        
        # Also show theoretical predictions for comparison (optional)
        z_theory, w_A_theory, w_B_theory = self.engine.simulate_b_mode_resolution(fixed_depth)
        ax4.plot(w_A_theory * 1000, z_theory * 1000, 'r--', alpha=0.5, label='Case A: Theory', linewidth=1)
        ax4.plot(w_B_theory * 1000, z_theory * 1000, 'b--', alpha=0.5, label='Case B: Theory', linewidth=1)

        ax4.set_ylim(100, 0)
        ax4.set_xlim(0, max(np.max(w_A_mm), np.max(w_B_mm), np.max(w_A_theory*1000), np.max(w_B_theory*1000)) * 1.2)
        ax4.set_xlabel("Beam Width (mm)", fontsize=9)
        ax4.set_ylabel("Depth (mm)", fontsize=9)
        ax4.set_title('Measured Resolution\nvs Depth', fontsize=9, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=6)
        ax4.tick_params(labelsize=8)
        
        # Mark the fixed focus depth on resolution plot
        ax4.axhline(fixed_depth * 1000, color='red', linestyle='--', alpha=0.5)
        
        # Add DOF comparison as text annotations
        print("Calculating Depth of Field...")
        dof_A_range, dof_A_start, dof_A_end = self.engine.calculate_depth_of_field(fixed_depth, 'fixed')
        dof_B_range, dof_B_start, dof_B_end = self.engine.calculate_depth_of_field(fixed_depth, 'dynamic')
        
        # Show DOF zones on images
        # Case A: Show limited DOF zone
        ax2.axhspan(dof_A_start*1000, dof_A_end*1000, alpha=0.15, color='green', 
                   label=f'DOF: {dof_A_range*1000:.1f}mm')
        
        # Case B: Show extended DOF zone (nearly full range)
        ax3.axhspan(dof_B_start*1000, dof_B_end*1000, alpha=0.1, color='green',
                   label=f'DOF: {dof_B_range*1000:.1f}mm')
        
        # Add DOF comparison text
        dof_text = (f"Depth of Field Comparison:\n"
                   f"Case A (Fixed): {dof_A_range*1000:.1f} mm\n"
                   f"Case B (Dynamic): {dof_B_range*1000:.1f} mm\n"
                   f"Improvement: {dof_B_range/dof_A_range:.1f}x")
        
        ax4.text(0.02, 0.02, dof_text, transform=ax4.transAxes, fontsize=7, 
                va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

        # Add comparison annotations
        ax2.text(0.02, 0.98, f'Notice: Blurring\naway from focus line\n\nDOF: {dof_A_range*1000:.1f}mm\n(Green shaded zone)', 
                transform=ax2.transAxes, fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax3.text(0.02, 0.98, f'Notice: Consistent\nquality at all depths\n\nDOF: {dof_B_range*1000:.1f}mm\n(Nearly full range)', 
                transform=ax3.transAxes, fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

        self.fig2.tight_layout(pad=1.0)
        self.canvas2.draw()
        print("Complete imaging analysis generated successfully!")


if __name__ == '__main__':
    plt.rcdefaults()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())