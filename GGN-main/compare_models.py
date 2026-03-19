# compare_models.py（放项目根目录）
import os, json, glob
import numpy as np

from eeg_main import (  # 直接复用我们刚才写的绘图函数
    plot_compare_bars, plot_per_class_f1, plot_macro_roc_pr, plot_calibration
)

def load_one(mjson):
    with open(mjson,'rt') as f:
        md = json.load(f)
    tag = os.path.basename(mjson).replace("_metrics.json","")
    npz = mjson.replace("_metrics.json","_test_proba_labels.npz")
    d = np.load(npz)
    return tag, md, d["y_true"], d["proba"]

if __name__ == "__main__":
    # 指定某个日期目录，比如 ./figs/20251022/ 下存放了三组文件
    date_dir = "./figs/{}".format(input("date dir (e.g., 20251022): ").strip())
    os.makedirs(date_dir, exist_ok=True)

    metrics_files = glob.glob(os.path.join(date_dir, "*_metrics.json"))
    metrics_dict = {}
    proba_dict   = {}
    y_true_any   = None

    for mjs in metrics_files:
        name, md, y_true, proba = load_one(mjs)
        # 统一模型别名（可根据你的实际命名）
        # ############################################################
        # if "ST-HGGN" in name: disp = "ST-HGGN"
        # # elif "ST-HGGN" in name or "ST_HGGN" in name: disp = "ST-HGGN"
        # elif "HGGN" in name: disp = "HGGN"
        # elif "GGN" in name: disp = "GGN"
        # elif "NeuroDynamicNet" in name: disp = "NeuroDynamicNet"
        # else: disp = name
        # #############################################################
        if "cnnnet" in name: disp = "cnnnet"
        elif "gnnnet" in name: disp = "gnnnet"
        elif "transformer" in name: disp = "transformer"
        elif "NeuroDynamicNet" in name: disp = "NeuroDynamicNet"
        else: disp = name
        metrics_dict[disp] = md
        proba_dict[disp]   = proba
        y_true_any = y_true

    # 画图
    plot_compare_bars(date_dir, metrics_dict, title="EEG models (macro)")
    plot_per_class_f1(date_dir, metrics_dict)
    # 取 label_names（从任意 json 里读）
    label_names = [pc["label"] for pc in next(iter(metrics_dict.values()))["per_class"]]
    plot_macro_roc_pr(date_dir, y_true_any, proba_dict, label_names)
    plot_calibration(date_dir, y_true_any, proba_dict)

    print("Done. See figures under:", date_dir)
