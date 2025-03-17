import mujoco
import os
from pathlib import Path

try:
    # Load our warehouse model
    model_path = os.path.join(Path(__file__).parent, "assets", "warehouse.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Perform a simulation step
    mujoco.mj_step(model, data)

    # Print some data to verify
    print("MuJoCo Python bindings are working!")
    print(f"Time: {data.time}")
    print(f"Robot position: {data.qpos[:2]}")
    print(f"Robot orientation: {data.qpos[2]}")

except Exception as e:
    print(f"Error: MuJoCo Python bindings are not working. {e}")