python3 main.py --server_address localhost:4445 \
--rounds 3 --fraction_fit 1 --fraction_eval 1 \
--min_sample_size 2 --local_epochs 2 --load_weights \
--model HugoNet --n_classes 3 --batch_size 32 \
--mlflow_server_ip localhost --mlflow_server_port 4040

# TODO MQTT Canal to control the server including 
# changing parameters, models, 
# starting experiment
# TODO Put db to manage models
