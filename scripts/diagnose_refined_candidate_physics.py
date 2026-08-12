#!/usr/bin/env python3
"""Run model/mode ablation at the best fine-HFSS length-refined candidate."""

import validate_hfss_corrected_extended_models as validation

validation.SELECTED = {
    "pump_GHz": 12.70,
    "Idc_A": 360e-6,
    "Ip0_A": 280e-6,
    "supercell_repeats": 1050,
}
validation.NEARBY_PUMPS_GHZ = (12.68, 12.70, 12.72)
validation.OUTPUT_STEP = "step_18_refined_candidate_diagnosis"

if __name__ == "__main__":
    validation.main()
