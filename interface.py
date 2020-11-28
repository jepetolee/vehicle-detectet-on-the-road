import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('대포차 검진기')
        self.move(300, 300)
        self.resize(1280, 800)
        btn2 = QPushButton(self)
        btn2.setText('Button&2')
        vbox = QVBoxLayout()
        vbox.addWidget(btn2)
        self.show()


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())