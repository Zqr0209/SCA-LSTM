from pathlib import Path


class AnalysisConfig:
    project_root = Path(__file__).absolute().parent.parent
    # benchmark_root = Path("E:/benchmark_models/netcdf")
    # baseline = "pub_lstm"
    # baseline_root = "pub_lstm_nldas_runs"
    # baseline = "benchmark_nwm_retrospective"
    # baseline_root = "benchmark_nwm_retrospective_runs"
    # baseline = "benchmark_sacsma_ensemble"
    # baseline_root = "benchmark_sacsma_ensemble_runs"

    # analyse_root = project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333" \
    #                / "pretrain_test_single"
    # analyse_root = project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333" \
    #                / "fine_tune"

    # analyse_root = project_root / "runs" / "TransformerUnPretrained_NAR_pt0[64-4-4-256-0.1]@daymet673@15+7[5+1]@NSELoss_n200_bs512_lr0.001_1980~1995#1995~2000#2000~2014@seed2333" \
    #                / "fine_tune"
    # analyse_root = project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet671_1980~1995#1995~2000#2000~2014@22|15+7[32+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333" \
    #                / "fine_tune"

    # analyse_root = project_root / "runs" / "baseline_lstm_s2s"
    # analyse_root = project_root / "runs" / "baseline_lstm_msv_s2s"

    # global models
    # analyse_root = project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@maurer_extended448_2001~2008#1999~2001#1989~1999@22|15+7[32+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333" \
    #                / "pretrain_test_single"
    # analyse_root = project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@maurer_extended448_2001~2008#1999~2001#1989~1999@15+7[5+1]@NSELoss_n200_bs512_lr0.001@seed2333" \
    #                / "pretrain_test_single"
    # analyse_root = project_root / "runs" / "LSTMMSVS2S_[[22-7-32-50][15-*-1-20][*-7-70-50]-0.2]@maurer_extended448_2001~2008#1999~2001#1989~1999@22|15+7[32+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed2333" \
    #                / "pretrain_test_single"
    # analyse_root = project_root / "runs" / "withstatic_Transformer_NAR_pt0[64-4-4-256-0.1]@maurer_ext448_01~08#99~01#89~99"

    # analyse_root = project_root / "runs" / "withstatic_lstm_ensemblel@maurer_ext448from531_01~08#99~01#89~99"
    # analyse_root = project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@maurer_extended448_2001~2008#1999~2001#1989~1999@22|15+7[32+1]@NSELoss_n300_bs512_lr0.001_warm_up@seed2333" \
    #                 / "pretrain_test_single"
    # analyse_root = project_root / "runs" / "withstatic_Transformer_NAR_pt0[64-4-4-256-0.1]@maurer_ext448_01~08#99~01#89~99_test" \
    #                 / "pretrain_test_single"

    # 365+365
    # analyse_root = project_root / "runs" / "Transformer_NAR_pt0[64-4-4-256-0.1]@daymet673_1980~1995#1995~2000#2000~2014@365+365[5+1]@MSE_n50_bs32_lr6.25e-05@seed2333" \
    #                / "fine_tune"

    # analyse_root = project_root / "runs/AirQua/Transformer_NAR_pt0[64-1-6-256-0.3]@AirQua_2014-01-01~2015-12-31#2016-01-01~2016-12-31#2017-01-01~2017-12-31@15+1[14+1]@NSELoss_n500_bs256_lr0.0005_exp_decay@seed2333"
    analyse_root = project_root / "runs" /"normal"/ "month_LSTMATTN_None_[[8-1-8-128][*-1-8-128]-0.2]@114_AUS_basins_list_1977~2006#2007~2009#2010~2014@8|7+1[8+1]@NSELoss_n200_bs32_lr0.001_warm_up@seed2333"
