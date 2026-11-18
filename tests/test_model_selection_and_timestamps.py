import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_eval import cmu_browser_password_capture, select_sentence_encoder
from preprocessing import (
    KEYRECS_DD_COLS,
    KEYRECS_FIXED_COLUMN_MAPS,
    KEYRECS_HOLD_COLS,
    audit_keyrecs_fixed_timestamps,
    read_keyrecs_fixed_timings,
)


class TestModelSelectionAndTimestamps(unittest.TestCase):
    def test_fixed_timing_mapping_is_selected_from_train_and_reused_for_other_splits(self):
        holds = np.linspace(0.08, 0.17, 10)
        dd = np.linspace(0.19, 0.31, 9)
        physical = {
            "DD": dd,
            "DU": dd + holds[1:],
            "UD": dd - holds[:-1],
            "UU": dd + holds[1:] - holds[:-1],
        }

        def row(participant, interpretation):
            record = {"participant": participant, "session": 1}
            record.update(zip(KEYRECS_HOLD_COLS, holds))
            for timing, prefix in KEYRECS_FIXED_COLUMN_MAPS[interpretation].items():
                for name, value in zip(KEYRECS_DD_COLS, physical[timing]):
                    record[f"{prefix}.{name.partition('.')[2]}"] = value
            return record

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.csv"
            pd.DataFrame([row("train", "release_label_swap"), row("val", "published_labels"),
                          row("test", "published_labels")]).to_csv(path, index=False)
            audit = audit_keyrecs_fixed_timestamps(path, ["train"])
            self.assertEqual(audit["selected_interpretation"], "release_label_swap")
            self.assertEqual(audit["train_rows"], 1)
            for name, expected in (("train", physical["DD"]), ("val", physical["UU"])):
                _, raw_h, raw_d, raw_u = read_keyrecs_fixed_timings(path, [name], audit["selected_interpretation"])
                np.testing.assert_allclose(raw_d[0], expected, atol=1e-7)
                np.testing.assert_allclose(raw_u, raw_d - raw_h[:, :-1], atol=1e-7)

    def test_browser_sample_excludes_terminal_key_before_extracting_features(self):
        holds = [0.09] * 10 + [20.0]
        dds = [0.17] * 9 + [35.0]
        cap = cmu_browser_password_capture(holds, dds)
        self.assertEqual(cap["holds"], holds[:10])
        self.assertEqual(cap["clean_digraphs"], 9)
        self.assertEqual(cap["pair_keys"][-1], ["KeyN", "KeyL"])
        np.testing.assert_allclose(cap["dds"], dds[:9])

    def test_clean_cross_session_top1_selects_either_encoder_and_breaks_ties(self):
        scores = {"keyprint_17ch": {"top1_accuracy": 0.6}, "vanilla_17ch_aug": {"top1_accuracy": 0.7}}
        self.assertEqual(select_sentence_encoder(scores), "vanilla_17ch_aug")
        scores["keyprint_17ch"]["top1_accuracy"] = 0.7
        self.assertEqual(select_sentence_encoder(scores), "keyprint_17ch")


if __name__ == "__main__":
    unittest.main()
