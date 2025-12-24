import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QGroupBox, QTabWidget)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ==================================================================================
# TASK 8: DYNAMIC VS FIXED RECEIVE FOCUSING SIMULATION
# ==================================================================================

# --- PHYSICS CONSTANTS ---
C = 1540.  # Speed of sound in tissue (m/s)


class UltrasoundPhysicsEngine:
    def __init__(self):
        """
        SETUP: Defines the hardware and simulation environment.
        """
        self.num_elements = 64
        self.pitch = 0.3e-3
        width = (self.num_elements - 1) * self.pitch
        self.xi = np.linspace(-width / 2, width / 2, self.num_elements)
        self.fc = 5.0e6
        self.active_elements = 64

        # Grid for Tab 1 (Beam Map Visualization)
        self.nz = 200
        self.nx = 150
        self.z_max = 100e-3
        self.x_max = 20e-3
        z_axis = np.linspace(1e-3, self.z_max, self.nz)
        x_axis = np.linspace(-self.x_max, self.x_max, self.nx)
        self.X_grid, self.Z_grid = np.meshgrid(x_axis, z_axis)

        # --- B-MODE IMAGING PARAMETERS (High Res to avoid aliasing) ---
        self.n_lines = 200  # High lateral resolution (catch thin beams)
        self.n_samples = 400  # High axial resolution (catch thin pulses)
        self.scan_range = 15e-3  # +/- 15mm
        self.depth_range = 100e-3

        # Generate the Grid Phantom
        self.generate_scatterer_field()

    # ==============================================================================
    # VISUALIZATION GENERATORS (Tab 1)
    # ==============================================================================
    def calculate_single_beam_field(self, focal_depth):
        k = 2 * np.pi * self.fc / C
        dist_focus_to_element = np.sqrt(self.xi ** 2 + focal_depth ** 2)
        dist_focus_center = focal_depth

        # Focusing Phase
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

        delays = path_diff / C
        return np.clip(img_db, -60, 0), delays

    # ==============================================================================
    # ANALYSIS & METRICS
    # ==============================================================================
    def calculate_theoretical_width(self, z_depths, focus_depth, strategy='fixed'):
        """
        Calculates continuous beam width curve for plotting the 'V' shape.
        """
        D = self.active_elements * self.pitch
        lam = C / self.fc
        widths = []

        for z in z_depths:
            if strategy == 'dynamic':
                # Dynamic: Always focused at current depth z
                w = (z * lam) / D
            else:
                # Fixed: Focused at focus_depth, diverging elsewhere
                w0 = (focus_depth * lam) / D
                defocus = abs(z - focus_depth)
                if z < focus_depth:
                    w = w0 + (defocus / focus_depth) * (D - w0)
                else:
                    divergence = D / focus_depth
                    w = w0 + defocus * divergence
            widths.append(w)

        return np.array(widths)

    def generate_scatterer_field(self):
        """
        THE PHANTOM: Creates a Grid of dots to see symmetry and fading.
        """
        # Create a Grid
        x_positions = np.linspace(-10e-3, 10e-3, 5)  # 5 Cols
        z_positions = np.linspace(20e-3, 90e-3, 8)  # 8 Rows
        xx, zz = np.meshgrid(x_positions, z_positions)

        self.scat_x = xx.flatten()
        self.scat_z = zz.flatten()

        # Constant Amplitude = 1.0 (To test pure attenuation)
        self.scat_amp = np.ones_like(self.scat_x)

    # ==============================================================================
    # SIMULATION PIPELINE
    # ==============================================================================
    def simulate_bmode_line(self, beam_x, focus_strategy, focus_depth=None):
        k = 2 * np.pi * self.fc / C
        z_samples = np.linspace(5e-3, self.depth_range, self.n_samples)
        line_intensity = np.zeros(self.n_samples)

        for i, z_sample in enumerate(z_samples):
            # Focus Switch
            if focus_strategy == 'fixed':
                focus_z = focus_depth
            else:
                focus_z = z_sample

                # 1. Steered Focusing Phase (Corrected Logic)
            dist_focus = np.sqrt((self.xi - beam_x) ** 2 + focus_z ** 2)
            dist_ref = np.sqrt(beam_x ** 2 + focus_z ** 2)
            focusing_phase = -k * (dist_focus - dist_ref)

            # 2. Select Scatterers
            cell_width = self.calculate_beam_width(z_sample, focus_z)
            cell_height = C / (2 * self.fc)

            x_mask = np.abs(self.scat_x - beam_x) < cell_width / 2
            z_mask = np.abs(self.scat_z - z_sample) < cell_height / 2
            active_mask = x_mask & z_mask

            if np.any(active_mask):
                scat_x_active = self.scat_x[active_mask]
                scat_z_active = self.scat_z[active_mask]
                scat_amp_active = self.scat_amp[active_mask]

                # 3. Coherent Summation
                dx = scat_x_active[np.newaxis, :] - self.xi[:, np.newaxis]
                dz = scat_z_active[np.newaxis, :]
                distances = np.sqrt(dx ** 2 + dz ** 2)

                receive_phases = k * distances + focusing_phase[:, np.newaxis]
                receive_signals = scat_amp_active[np.newaxis, :] * np.exp(1j * receive_phases)
                coherent_sum = np.sum(receive_signals, axis=0)

                total_amplitude = np.abs(np.sum(coherent_sum))

                # 4. Apply Physics (Attenuation ONLY)
                attenuation_factor = self.calculate_attenuation(z_sample, self.fc)
                line_intensity[i] = total_amplitude * attenuation_factor

        return z_samples, line_intensity

    def calculate_beam_width(self, target_depth, focus_depth):
        D = self.active_elements * self.pitch
        lam = C / self.fc
        if abs(target_depth - focus_depth) < 1e-6: return (focus_depth * lam) / D

        w0 = (focus_depth * lam) / D
        defocus = abs(target_depth - focus_depth)
        if target_depth < focus_depth:
            width = w0 + (defocus / focus_depth) * (D - w0)
        else:
            width = w0 + defocus * (D / focus_depth)
        return width

    def calculate_attenuation(self, depth, frequency):
        attenuation_coeff = 1.0
        depth_cm = depth * 100
        freq_mhz = frequency / 1e6
        total_loss_db = attenuation_coeff * depth_cm * freq_mhz
        return 10 ** (-total_loss_db / 20)

    def generate_bmode_image(self, focus_strategy, focus_depth=None):
        x_lines = np.linspace(-self.scan_range, self.scan_range, self.n_lines)
        bmode_image = np.zeros((self.n_samples, self.n_lines))

        for line_idx, x_beam in enumerate(x_lines):
            z_samples, line_data = self.simulate_bmode_line(x_beam, focus_strategy, focus_depth)
            bmode_image[:, line_idx] = line_data

        bmode_image = bmode_image / (np.max(bmode_image) + 1e-12)
        bmode_db = 20 * np.log10(bmode_image + 1e-6)
        bmode_db = np.clip(bmode_db, -60, 0)

        return x_lines, z_samples, bmode_db

    def generate_ground_truth_image(self):
        # Creates a perfect reference image of the grid
        x_gt = np.linspace(-self.scan_range, self.scan_range, self.n_lines)
        z_gt = np.linspace(5e-3, self.depth_range, self.n_samples)
        X_gt, Z_gt = np.meshgrid(x_gt, z_gt)
        gt_image = np.zeros_like(X_gt)

        for i in range(len(self.scat_x)):
            x_idx = np.argmin(np.abs(x_gt - self.scat_x[i]))
            z_idx = np.argmin(np.abs(z_gt - self.scat_z[i]))
            sigma_x = 0.3e-3
            sigma_z = 0.2e-3

            # Simple Gaussian blobs
            for dx in range(-3, 4):
                for dz in range(-3, 4):
                    xi = x_idx + dx
                    zi = z_idx + dz
                    if 0 <= xi < len(x_gt) and 0 <= zi < len(z_gt):
                        dist = ((x_gt[xi] - self.scat_x[i]) ** 2 + (z_gt[zi] - self.scat_z[i]) ** 2)
                        gt_image[zi, xi] += np.exp(-dist / (2 * sigma_x ** 2))

        gt_db = 20 * np.log10(gt_image + 1e-12)
        return x_gt, z_gt, np.clip(gt_db, -60, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Task 8: Ultrasound Simulation Suite")
        self.resize(1400, 900)

        self.engine = UltrasoundPhysicsEngine()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab1 = QWidget()
        self.setup_tab1()
        self.tabs.addTab(self.tab1, "Tab 1: Physics Lab")

        self.tab2 = QWidget()
        self.setup_tab2()
        self.tabs.addTab(self.tab2, "Tab 2: B-Mode Comparison")

    def setup_tab1(self):
        layout = QHBoxLayout(self.tab1)
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
        ax1.set_title("Beam Profile")
        ax1.plot(0, depth * 1000, 'rx', markersize=10)
        ax2 = self.fig1.add_subplot(122)
        ax2.plot(self.engine.xi * 1000, delays * 1e6, 'c')
        ax2.set_title("Delay Curve")
        self.canvas1.draw()

    def setup_tab2(self):
        layout = QHBoxLayout(self.tab2)
        controls = QVBoxLayout()

        info = QLabel(
            "<b>Analysis Guide:</b><br><br>"
            "<b>1. Ground Truth:</b> Actual dots.<br>"
            "<b>2. Case A (Fixed):</b> Notice the V-shape blur. Sharp only at focus line.<br>"
            "<b>3. Case B (Dynamic):</b> Sharp everywhere.<br>"
            "<b>4. Graph (Right):</b> Compares Resolution vs Depth.<br>"
            " - Red Curve: Case A (V-Shape)<br>"
            " - Blue Curve: Case B (Straight)<br>"
            " - Green Zone: The Depth of Field"
        )
        info.setWordWrap(True)
        controls.addWidget(info)

        grp = QGroupBox("Controls")
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Fixed Focus Depth:"))
        self.slider_focus_t2 = QSlider(Qt.Horizontal)
        self.slider_focus_t2.setRange(20, 80)
        self.slider_focus_t2.setValue(40)
        self.slider_focus_t2.valueChanged.connect(self.update_tab2)
        vbox.addWidget(self.slider_focus_t2)
        self.lbl_focus_t2 = QLabel("40 mm")
        vbox.addWidget(self.lbl_focus_t2)

        vbox.addWidget(QLabel("Frequency (MHz):"))
        self.slider_freq = QSlider(Qt.Horizontal)
        self.slider_freq.setRange(25, 120)
        self.slider_freq.setValue(50)
        self.slider_freq.valueChanged.connect(self.update_tab2)
        vbox.addWidget(self.slider_freq)
        self.lbl_freq = QLabel("5.0 MHz")
        vbox.addWidget(self.lbl_freq)

        grp.setLayout(vbox)
        controls.addWidget(grp)
        controls.addStretch()
        layout.addLayout(controls, 1)

        self.fig2 = Figure(figsize=(12, 6))
        self.canvas2 = FigureCanvas(self.fig2)
        layout.addWidget(self.canvas2, 3)

        self.update_tab2()

    def update_tab2(self):
        fixed_depth = self.slider_focus_t2.value() * 1e-3
        frequency = self.slider_freq.value() / 10.0 * 1e6

        self.lbl_focus_t2.setText(f"{fixed_depth * 1000:.0f} mm")
        self.lbl_freq.setText(f"{frequency / 1e6:.1f} MHz")
        self.engine.fc = frequency

        self.fig2.clear()

        # 1. Generate Images
        print("Generating images...")
        x_gt, z_gt, gt_img = self.engine.generate_ground_truth_image()
        x_A, z_A, bmode_A = self.engine.generate_bmode_image('fixed', fixed_depth)
        x_B, z_B, bmode_B = self.engine.generate_bmode_image('dynamic')

        # 2. Setup Layout (4 Columns)
        ax1 = self.fig2.add_subplot(141)
        ax2 = self.fig2.add_subplot(142)
        ax3 = self.fig2.add_subplot(143)
        ax4 = self.fig2.add_subplot(144)  # THE NEW PLOT

        # 3. Plot Images
        extent = [-15, 15, 100, 0]
        ax1.imshow(gt_img, extent=extent, cmap='hot', aspect='auto', vmin=-60, vmax=0)
        ax1.set_title("Ground Truth")

        ax2.imshow(bmode_A, extent=extent, cmap='gray', aspect='auto', vmin=-60, vmax=0)
        ax2.set_title("Case A: Fixed")
        ax2.axhline(fixed_depth * 1000, color='red', linestyle='--')

        ax3.imshow(bmode_B, extent=extent, cmap='gray', aspect='auto', vmin=-60, vmax=0)
        ax3.set_title("Case B: Dynamic")

        # 4. PLOT 4: Theoretical Beam Width & DOF Comparison
        #    This replaces the old scattered dots with clean curves

        z_smooth = np.linspace(0, 100, 100) * 1e-3
        w_A = self.engine.calculate_theoretical_width(z_smooth, fixed_depth, 'fixed') * 1000
        w_B = self.engine.calculate_theoretical_width(z_smooth, fixed_depth, 'dynamic') * 1000
        z_mm = z_smooth * 1000

        # Plot Curves
        ax4.plot(w_A, z_mm, 'r-', linewidth=2, label='Case A (Fixed)')
        ax4.plot(w_B, z_mm, 'b-', linewidth=2, label='Case B (Dynamic)')

        # Calculate & Visualize DOF
        # Define DOF as region where beam width is < 1.5x the minimum width
        w_min = np.min(w_A)
        threshold = w_min * 1.5

        # Find start/end of DOF for Case A
        valid_indices = np.where(w_A < threshold)[0]
        if len(valid_indices) > 0:
            dof_top = z_mm[valid_indices[0]]
            dof_bot = z_mm[valid_indices[-1]]

            # Shade the DOF region
            ax4.axhspan(dof_top, dof_bot, color='red', alpha=0.1, label='Case A DOF')

            # Add Arrows/Text for DOF
            ax4.annotate(f"DOF: {dof_bot - dof_top:.1f}mm",
                         xy=(w_min, fixed_depth * 1000),
                         xytext=(w_min + 1.0, fixed_depth * 1000),
                         arrowprops=dict(facecolor='black', arrowstyle='->'))

        ax4.axvline(threshold, color='k', linestyle=':', alpha=0.5, label='Acceptable Limit')

        ax4.set_ylim(0, 100)
        ax4.set_xlim(0, max(np.max(w_A), 3.0))
        ax4.set_title("Lateral Res vs Depth")
        ax4.set_xlabel("Beam Width (mm)")
        ax4.set_ylabel("Depth (mm)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        self.fig2.tight_layout()
        self.canvas2.draw()


if __name__ == '__main__':
    plt.rcdefaults()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())