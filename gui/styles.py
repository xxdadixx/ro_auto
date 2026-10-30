class LiquidGlassTheme:
    """Native QSS (Qt Style Sheet) rules for the Liquid Glass Dark Mode theme."""

    MAIN_QSS = """
    /* Main Window & Base Background */
    QMainWindow, QDialog {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                    stop:0 #090d16, stop:0.5 #0d1322, stop:1 #060911);
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
    }

    /* Glass Panels / Cards */
    QFrame#glassPanel, QGroupBox#glassPanel {
        background-color: rgba(20, 27, 45, 0.65);
        border: 1px solid rgba(0, 242, 254, 0.18);
        border-radius: 12px;
    }

    /* Headings */
    QLabel#headerTitle {
        color: #ffffff;
        font-size: 18px;
        font-weight: bold;
    }

    QLabel#subText {
        color: #94a3b8;
        font-size: 12px;
    }

    /* Buttons */
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                    stop:0 rgba(0, 198, 255, 0.15), stop:1 rgba(0, 114, 255, 0.25));
        border: 1px solid rgba(0, 198, 255, 0.4);
        border-radius: 6px;
        color: #e0f2fe;
        font-weight: 600;
        padding: 6px 14px;
        font-size: 12px;
    }

    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                    stop:0 rgba(0, 198, 255, 0.35), stop:1 rgba(0, 114, 255, 0.45));
        border: 1px solid #00f2fe;
        color: #ffffff;
    }

    QPushButton:pressed {
        background: rgba(0, 114, 255, 0.6);
    }

    QPushButton#dangerButton {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
    }

    QPushButton#dangerButton:hover {
        background: rgba(239, 68, 68, 0.4);
        border: 1px solid #ef4444;
        color: #ffffff;
    }

    /* Tab Widget */
    QTabWidget::pane {
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 21, 37, 0.5);
        border-radius: 10px;
    }

    QTabBar::tab {
        background: rgba(20, 27, 45, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        padding: 8px 18px;
        margin-right: 4px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        font-weight: bold;
    }

    QTabBar::tab:selected {
        background: rgba(0, 242, 254, 0.15);
        border: 1px solid rgba(0, 242, 254, 0.4);
        color: #00f2fe;
    }

    /* Inputs & SpinBoxes */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background: rgba(10, 15, 28, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 6px;
        color: #f1f5f9;
        padding: 5px 10px;
    }

    QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
        border: 1px solid #00f2fe;
    }

    /* Log Output Terminal */
    QTextEdit#terminalLog {
        background-color: rgba(6, 9, 15, 0.9);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 8px;
        color: #38bdf8;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 11px;
    }

    /* Scrollbars */
    QScrollBar:vertical {
        background: rgba(10, 15, 28, 0.5);
        width: 8px;
        border-radius: 4px;
    }

    QScrollBar::handle:vertical {
        background: rgba(0, 242, 254, 0.3);
        border-radius: 4px;
    }

    QScrollBar::handle:vertical:hover {
        background: rgba(0, 242, 254, 0.6);
    }
    """