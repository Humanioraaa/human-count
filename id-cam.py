import win32com.client

def list_camera_devices():
    wmi = win32com.client.GetObject("winmgmts:")
    cameras = []
    for cam in wmi.InstancesOf("Win32_PnPEntity"):
        if "camera" in str(cam.Name).lower():
            cameras.append(cam.Name)
    return cameras

cameras = list_camera_devices()
if not cameras:
    print("No camera devices found.")
else:
    for cam in cameras:
        print(f"Camera Name: {cam}")
