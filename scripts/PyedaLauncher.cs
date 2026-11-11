using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

public static class PyedaLauncher
{
    [STAThread]
    public static void Main()
    {
        string pythonw = @"D:\program\EDA\.venv\Scripts\pythonw.exe";
        string gui = @"D:\program\EDA\scripts\pyeda_gui.py";
        if (!File.Exists(pythonw) || !File.Exists(gui))
        {
            MessageBox.Show("PYEDA GUI 文件不存在。\n" + pythonw + "\n" + gui, "PYEDA 启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }
        var info = new ProcessStartInfo
        {
            FileName = pythonw,
            Arguments = "\"" + gui + "\"",
            WorkingDirectory = @"D:\program\EDA",
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden
        };
        Process.Start(info);
    }
}
