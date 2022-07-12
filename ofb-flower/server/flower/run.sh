python3 main.py --server_address localhost:4050 \
--rounds 3 --fraction_fit 1 --fraction_eval 1 \
--min_sample_size 2 --local_epochs 2 --load_weights \
--model hugonet --n_classes 3 --batch_size 32 \
--mlflow_server_ip 127.0.0.1 --mlflow_server_port 5000

# TODO MQTT Canal to control the server including 
# changing parameters, models, 
# starting experiment
# TODO Put db to manage models

