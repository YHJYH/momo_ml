
class ModelMonitor:
    def __init__(self, ref_df, cur_df, label_col=None, pred_col=None):
        self.ref_df = ref_df
        self.cur_df = cur_df
        self.label_col = label_col
        self.pred_col = pred_col

    def run_all(self):
        return {"status": "ok"}
