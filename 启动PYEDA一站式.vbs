Option Explicit

Dim fso, shell, root, app, pythonw, gui
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)

app = fso.BuildPath(root, "PYEDA_一站式分析.exe")
If Not fso.FileExists(app) Then app = fso.BuildPath(root, "dist\PYEDA_一站式分析.exe")
If fso.FileExists(app) Then
    shell.CurrentDirectory = root
    shell.Run """" & app & """", 0, False
    WScript.Quit 0
End If

pythonw = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
gui = fso.BuildPath(root, "scripts\pyeda_gui.py")
If Not fso.FileExists(pythonw) Then pythonw = "D:\program\EDA\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(gui) Then gui = "D:\program\EDA\scripts\pyeda_gui.py"

If Not fso.FileExists(pythonw) Or Not fso.FileExists(gui) Then
    MsgBox "找不到 PYEDA 图形程序或 Python 环境。请把发布包解压完整，或确认 D:\program\EDA\.venv 存在。", vbCritical, "PYEDA 启动失败"
    WScript.Quit 1
End If

shell.CurrentDirectory = fso.GetParentFolderName(gui)
shell.Run """" & pythonw & """ """" & gui & """", 0, False
