from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

LABEL_MAPPING = {
    "background": (),  # label does not represent the outside of the whole body
    "spleen": ("body", "abdominal_cavity", "spleen"),
    "kidney_right": ("body", "abdominal_cavity", "kidneys", "kidney_right"),
    "kidney_left": ("body", "abdominal_cavity", "kidneys", "kidney_left"),
    "gallbladder": ("body", "abdominal_cavity", "gallbladder"),
    "liver": ("body", "abdominal_cavity", "liver"),
    "stomach": ("body", "abdominal_cavity", "stomach"),
    "aorta": (),  # only splitted labels are supported
    "inferior_vena_cava": (),  # only splitted labels are supported
    "portal_vein_and_splenic_vein": (
        "body",
        "abdominal_cavity",
        "portal_vein_and_splenic_vein",
    ),
    "pancreas": ("body", "abdominal_cavity", "pancreas"),
    "adrenal_gland_right": (
        "body",
        "abdominal_cavity",
        "adrenal_glands",
        "adrenal_gland_right",
    ),
    "adrenal_gland_left": (
        "body",
        "abdominal_cavity",
        "adrenal_glands",
        "adrenal_gland_left",
    ),
    "lung_upper_lobe_left": (
        "body",
        "thoracic_cavity",
        "lungs",
        "lung_left",
        "lung_upper_lobe_left",
    ),
    "lung_lower_lobe_left": (
        "body",
        "thoracic_cavity",
        "lungs",
        "lung_left",
        "lung_lower_lobe_left",
    ),
    "lung_upper_lobe_right": (
        "body",
        "thoracic_cavity",
        "lungs",
        "lung_right",
        "lung_upper_lobe_right",
    ),
    "lung_middle_lobe_right": (
        "body",
        "thoracic_cavity",
        "lungs",
        "lung_right",
        "lung_middle_lobe_right",
    ),
    "lung_lower_lobe_right": (
        "body",
        "thoracic_cavity",
        "lungs",
        "lung_right",
        "lung_lower_lobe_right",
    ),
    "vertebrae_L5": ("body", "bones", "spine", "lumbar_spine", "vertebrae_L5"),
    "vertebrae_L4": ("body", "bones", "spine", "lumbar_spine", "vertebrae_L4"),
    "vertebrae_L3": ("body", "bones", "spine", "lumbar_spine", "vertebrae_L3"),
    "vertebrae_L2": ("body", "bones", "spine", "lumbar_spine", "vertebrae_L2"),
    "vertebrae_L1": ("body", "bones", "spine", "lumbar_spine", "vertebrae_L1"),
    "vertebrae_T12": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T12"),
    "vertebrae_T11": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T11"),
    "vertebrae_T10": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T10"),
    "vertebrae_T9": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T9"),
    "vertebrae_T8": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T8"),
    "vertebrae_T7": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T7"),
    "vertebrae_T6": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T6"),
    "vertebrae_T5": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T5"),
    "vertebrae_T4": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T4"),
    "vertebrae_T3": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T3"),
    "vertebrae_T2": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T2"),
    "vertebrae_T1": ("body", "bones", "spine", "thoracic_spine", "vertebrae_T1"),
    "vertebrae_C7": ("body", "bones", "spine", "cervical_spine", "vertebrae_C7"),
    "vertebrae_C6": ("body", "bones", "spine", "cervical_spine", "vertebrae_C6"),
    "vertebrae_C5": ("body", "bones", "spine", "cervical_spine", "vertebrae_C5"),
    "vertebrae_C4": ("body", "bones", "spine", "cervical_spine", "vertebrae_C4"),
    "vertebrae_C3": ("body", "bones", "spine", "cervical_spine", "vertebrae_C3"),
    "vertebrae_C2": ("body", "bones", "spine", "cervical_spine", "vertebrae_C2"),
    "vertebrae_C1": ("body", "bones", "spine", "cervical_spine", "vertebrae_C1"),
    "esophagus": (),  # unsupported
    "trachea": (),  # unsupported
    "heart_myocardium": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "pericardium",
        "heart_myocardium",
    ),
    "heart_atrium_left": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "pericardium",
        "heart_atrium_left",
    ),
    "heart_ventricle_left": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "pericardium",
        "heart_ventricle_left",
    ),
    "heart_atrium_right": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "pericardium",
        "heart_atrium_right",
    ),
    "heart_ventricle_right": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "pericardium",
        "heart_ventricle_right",
    ),
    "pulmonary_artery": (),  # unsupported
    "brain": ("body", "brain"),
    "iliac_artery_left": (
        "body",
        "abdominal_cavity",
        "iliac_arteries",
        "iliac_artery_left",
    ),
    "iliac_artery_right": (
        "body",
        "abdominal_cavity",
        "iliac_arteries",
        "iliac_artery_right",
    ),
    "iliac_vena_left": ("body", "abdominal_cavity", "iliac_venae", "iliac_vena_left"),
    "iliac_vena_right": ("body", "abdominal_cavity", "iliac_venae", "iliac_vena_right"),
    "small_bowel": ("body", "abdominal_cavity", "small_bowel"),
    "duodenum": ("body", "abdominal_cavity", "duodenum"),
    "colon": ("body", "abdominal_cavity", "colon"),
    "rib_left_1": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_1"),
    "rib_left_2": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_2"),
    "rib_left_3": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_3"),
    "rib_left_4": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_4"),
    "rib_left_5": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_5"),
    "rib_left_6": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_6"),
    "rib_left_7": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_7"),
    "rib_left_8": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_8"),
    "rib_left_9": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_9"),
    "rib_left_10": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_10"),
    "rib_left_11": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_11"),
    "rib_left_12": ("body", "bones", "rib_cage", "rib_cage_left", "rib_left_12"),
    "rib_right_1": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_1"),
    "rib_right_2": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_2"),
    "rib_right_3": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_3"),
    "rib_right_4": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_4"),
    "rib_right_5": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_5"),
    "rib_right_6": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_6"),
    "rib_right_7": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_7"),
    "rib_right_8": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_8"),
    "rib_right_9": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_9"),
    "rib_right_10": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_10"),
    "rib_right_11": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_11"),
    "rib_right_12": ("body", "bones", "rib_cage", "rib_cage_right", "rib_right_12"),
    "humerus_left": ("body", "bones", "humeri", "humerus_left"),
    "humerus_right": ("body", "bones", "humeri", "humerus_right"),
    "scapula_left": ("body", "bones", "scapulae", "scapula_left"),
    "scapula_right": ("body", "bones", "scapulae", "scapula_right"),
    "clavicula_left": ("body", "bones", "claviculae", "clavicula_left"),
    "clavicula_right": ("body", "bones", "claviculae", "clavicula_right"),
    "femur_left": ("body", "bones", "femora", "femur_left"),
    "femur_right": ("body", "bones", "femora", "femur_right"),
    "hip_left": ("body", "bones", "hips", "hip_left"),
    "hip_right": ("body", "bones", "hips", "hip_right"),
    "sacrum": ("body", "bones", "sacrum"),
    "face": (),  # unsupported, preditions to imprecise
    "gluteus_maximus_left": (
        "body",
        "muscles",
        "gluteus_maximi",
        "gluteus_maximus_left",
    ),
    "gluteus_maximus_right": (
        "body",
        "muscles",
        "gluteus_maximi",
        "gluteus_maximus_right",
    ),
    "gluteus_medius_left": ("body", "muscles", "gluteus_medii", "gluteus_medius_left"),
    "gluteus_medius_right": (
        "body",
        "muscles",
        "gluteus_medii",
        "gluteus_medius_right",
    ),
    "gluteus_minimus_left": (
        "body",
        "muscles",
        "gluteus_minimi",
        "gluteus_minimus_left",
    ),
    "gluteus_minimus_right": (
        "body",
        "muscles",
        "gluteus_minimi",
        "gluteus_minimus_right",
    ),
    "autochthon_left": ("body", "muscles", "autochthone", "autochthon_left"),
    "autochthon_right": ("body", "muscles", "autochthone", "autochthon_right"),
    "iliopsoas_left": ("body", "muscles", "iliopsoai", "iliopsoas_left"),
    "iliopsoas_right": ("body", "muscles", "iliopsoai", "iliopsoas_right"),
    "urinary_bladder": ("body", "abdominal_cavity", "urinary_bladder"),
    # new labels
    "aorta_abdominalis": ("body", "abdominal_cavity", "aorta_abdominalis"),
    "aorta_thoracica_pass_mediastinum": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "aorta_thoracica_pass_mediastinum",
    ),
    "aorta_thoracica_pass_pericardium": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "pericardium",
        "aorta_thoracica_pass_pericardium",
    ),
    "vci_pass_abdominalis": ("body", "abdominal_cavity", "vci_pass_abdominalis"),
    "vci_pass_thoracica": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "vci_pass_thoracica",
    ),
    "pulmonary_artery_pass_mediastinum": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "pulmonary_artery_pass_mediastinum",
    ),
    "pulmonary_artery_pass_pericardium": (
        "body",
        "thoracic_cavity",
        "mediastinum",
        "pericardium",
        "pulmonary_artery_pass_pericardium",
    ),
}


def main(args: Namespace) -> None:
    with args.input_labels.open() as ifile:
        label_names = ifile.read().splitlines()

    mapped_label_names: List[str] = []
    for label_name in label_names:
        if label_name not in LABEL_MAPPING:
            raise ValueError(f"Unknown label name found: {label_name}")

        label_path = LABEL_MAPPING[label_name]
        if not label_path:
            mapped_label_names.append("-")
        else:
            mapped_label_names.append(",".join(label_path))

    with args.tree_labels.open("w") as ofile:
        ofile.write("\n".join(mapped_label_names))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-labels", type=Path)
    parser.add_argument("--tree-labels", type=Path)
    args = parser.parse_args()

    main(args)
