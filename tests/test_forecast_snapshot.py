def test_forecast_summary_snapshot(bvar, snapshot):
    bvar.forecast(H=4, N_draws=500, random_state=1234, format=True)
    csv = bvar.df_forecasts_unconditional.round(6).to_csv(index=False)
    assert csv == snapshot
