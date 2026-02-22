from pathlib import Path
import os
from tensorflow.keras import Model

# @Params 
#   model_name: The name for your model
#   version_increment_type: "major", "minor", "revision"
#   save_path: Path where you want to save your model
# output creates a directory with the model_name inside the save_path,
# then creates the latest version inside that created directory.
#
# format: model_name_vMajor.Minor.Revisions.file_type
def save_model(model: Model, model_name: str, save_path: Path, version_increment_type = "revision", file_type = ".h5"):
    if version_increment_type != "major" and version_increment_type != "minor" and version_increment_type != "revision":
        print(f"version_increment_type must be either major or minor or revision not \"{version_increment_type}\".")
        return

    save_path = Path(str(save_path.absolute()) + str(os.sep) + model_name)

    save_path.mkdir(parents=True, exist_ok=True)

    existing_models = ["".join(model.split(".")[:-1]) for model in save_path.iterdir() if model.suffix == file_type]

    highest_version = "1.0.0"
    final_version = highest_version

    # Get the highest version
    if len(existing_models) != 0:
        for model in existing_models:
            current_version = model.split("_v")[-1]
            # Model files that don't follow the format are skipped
            if len(current_version) != 2:
                continue
            highest_version = helper_max_version(current_version, highest_version)

        # Increment version
        versions = highest_version.split('.')
        major = int(versions[0])
        minor = int(versions[1])
        revisions = int(versions[2])

        if version_increment_type == "major":
            major += 1
        elif version_increment_type == "minor":
            minor += 1
        else:
            revisions += 1

        final_version = f"{major}.{minor}.{revisions}"

    model_full_name = f"{model_name}_v{final_version}"
    
    final_save_path = f"{str(save_path)}{os.sep}{model_full_name}{file_type}"

    model.save(final_save_path)

    return True

def helper_max_version(version_a: str, version_b: str):
    v_a = version_a.split(".")
    v_a_t = (v_a[0], v_a[1], v_a[2])

    v_b = version_b.split(".")
    v_b_t = (v_b[0], v_b[1], v_b[2])

    return version_a if v_a_t > v_b_t else version_b