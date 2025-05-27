import pypidtune

pid_tuner = pypidtune.PIDTuner(
    pid_type="pid",
    pid_form="dependent",
    pid_time_units="minutes",
    init_model_gain=3.0,
    init_model_tc=60.0,
    init_model_dt=15.0,
    init_model_bias=10.0,
    default_csv_sep=";",
)
pid_tuner.root.mainloop()
pid_tuner.tune_pid()
