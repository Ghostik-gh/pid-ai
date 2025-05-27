# PID imports
from PID_Py.PID import PID, ThreadedPID, HistorianParams
from PID_Py.SetupTool import SetupToolApp
from PID_Py.Simulation import Simulation

# PySide6 (PyQt) imports
from PySide6.QtWidgets import QApplication

import sys

# Threaded PID creation
pid = ThreadedPID(
    kp=1,
    ki=0,
    kd=0.0,
    cycleTime=0.1,
    historianParams=HistorianParams.SETPOINT
    | HistorianParams.PROCESS_VALUE
    | HistorianParams.OUTPUT
    | HistorianParams.ERROR,
    simulation=Simulation(1, 1),
)
pid.start()

# PyQt application creation
app = QApplication(sys.argv)

# SetupTool instantiation
setupToolApp = SetupToolApp(pid)
# setupToolApp.setReadWriteMode()
# setupToolApp.setpointControlSetEnabled(True)
setupToolApp.show()

# Application execution
app.exec()

# Application ended, stop the PID
pid.quit = True
pid.join()
