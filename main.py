import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QGroupBox, QTabWidget)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- PHYSICS CONSTANTS ---
C = 1540.0


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
            "<b>Task 8 Analysis:</b><br><br>"
            "Comparing imaging of 4 targets<br>"
            "(20, 40, 60, 80 mm).<br><br>"
            "<b>Case A (Left):</b><br>"
            "Fixed Focus (Set by slider).<br>"
            "Note the blurriness (Width) changes.<br><br>"
            "<b>Case B (Right):</b><br>"
            "Dynamic Focusing.<br>"
            "Focus tracks the depth.<br>"
            "Note constant high resolution."
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

        controls.addStretch()
        layout.addLayout(controls, 1)

        # Plots
        self.fig2 = Figure(figsize=(10, 8))
        self.canvas2 = FigureCanvas(self.fig2)
        layout.addWidget(self.canvas2, 3)

        self.update_tab2()

    def update_tab2(self):
        fixed_depth = self.slider_focus_t2.value() * 1e-3
        self.lbl_focus_t2.setText(f"{fixed_depth * 1000:.0f} mm")

        # Get Simulation Data
        z_targets, width_A, width_B = self.engine.simulate_b_mode_resolution(fixed_depth)

        self.fig2.clear()

        # --- VISUALIZATION: Simulated B-Mode ---
        # We draw "Blobs" to represent the scatterers based on calculated width
        ax_img = self.fig2.add_subplot(121)
        ax_img.set_facecolor('black')

        # Draw Case A Scatterers (Left side of plot)
        x_offset_A = -10
        for i, z in enumerate(z_targets):
            w = width_A[i] * 1000  # width in mm
            # Draw ellipse
            ellipse = plt.Circle((x_offset_A, z * 1000), w / 2, color='red', alpha=0.8)
            ax_img.add_patch(ellipse)

        # Draw Case B Scatterers (Right side of plot)
        x_offset_B = 10
        for i, z in enumerate(z_targets):
            w = width_B[i] * 1000
            ellipse = plt.Circle((x_offset_B, z * 1000), w / 2, color='cyan', alpha=0.8)
            ax_img.add_patch(ellipse)

        ax_img.set_xlim(-20, 20)
        ax_img.set_ylim(100, 0)
        ax_img.set_title("Simulated B-Mode Image")
        ax_img.text(x_offset_A, 5, "Case A\nFixed", color='red', ha='center', fontweight='bold')
        ax_img.text(x_offset_B, 5, "Case B\nDynamic", color='cyan', ha='center', fontweight='bold')

        # Draw the Fixed Focus Line for Case A reference
        ax_img.axhline(fixed_depth * 1000, color='red', linestyle='--', alpha=0.5)
        ax_img.text(-18, fixed_depth * 1000 - 2, "Fixed Focus", color='red', fontsize=8)

        # --- GRAPH: Lateral Resolution vs Depth ---
        ax_graph = self.fig2.add_subplot(122)
        z_mm = z_targets * 1000
        w_A_mm = width_A * 1000
        w_B_mm = width_B * 1000

        ax_graph.plot(w_A_mm, z_mm, 'r-o', label='Case A: Fixed', linewidth=2)
        ax_graph.plot(w_B_mm, z_mm, 'c-o', label='Case B: Dynamic', linewidth=2)

        ax_graph.set_ylim(100, 0)  # Depth on Y axis
        ax_graph.set_xlim(0, max(np.max(w_A_mm), np.max(w_B_mm)) * 1.2)
        ax_graph.set_xlabel("Lateral Resolution (Beam Width mm)")
        ax_graph.set_ylabel("Depth (mm)")
        ax_graph.set_title("Resolution vs Depth")
        ax_graph.grid(True)
        ax_graph.legend()

        self.fig2.tight_layout()
        self.canvas2.draw()


if __name__ == '__main__':
    plt.rcdefaults()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())