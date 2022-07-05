mlflow server -h 0.0.0.0 --backend-store-uri file:///Users/hugo/Documents/Programming/Orange/flower_federated_learning/ofb-flower/server/mlflow_storage
#TODO Sql server
#URI to which to persist experiment and run
#data. Acceptable URIs are SQLAlchemy-compatible
#database connection strings (e.g.
#'sqlite:///path/to/file.db') or local
#filesystem URIs (e.g.
# 'file:///absolute/path/to/directory'). By
#default, data will be logged to the ./mlruns
# directory. 
# TODO   --default-artifact-root URI -->  Directory in which to store artifacts for any new experiments created.
# For tracking server backends that rely on SQL, this option is required in order to store artifacts