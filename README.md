# TOD-DNN

Small dnn model to predict the time of death based on the work of "Henssge’s time of death estimation".<br>
https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0920-y<br>
<br>

<b>HOW TO USE:</b><br>
1. generate synthetic data with generate_synthetic_data_grid.py to custom directory<br>
2. calculate dataset related metrics with calc_norm_scalars.py<br>
3. run train script with given parameters (std and mean values are set to default testing results, must be overridden with args)<br>
<br>
inferencing.py uses the result_scalar param as upscaler, so this must be set to max_delta_time.
