# Database

The `src.db` package is the only code that talks to MongoDB. Everything downstream receives a `pymongo` `Database` handle from `get_database` and passes it to the loaders.

The dataset stores one document per mouse (`subjects`), per recording day (`dates`), and per trial (`trials`). Each trial's spike times and behavior channels live in GridFS as one zstd-compressed BSON document (`neurons_data`, `behavior_data`), which `get_neural` and `get_behavior` reassemble and decode into plain dicts. Sessions are addressed by name (`M<mouse id>_<YYMMDD>`), and `session_name_to_date_id` resolves a name to its `dates` document.

!!! note "Credentials"
    `get_client`, `get_database` and the loaders need a `.env` file (copied from `.env.example`) with `MONGODB_URI` and `MONGODB_DATABASE`. Add the `SSH_*` variables when the database is only reachable through an SSH tunnel. `session_name_to_display` works offline. The unit tests mock the client, so they need no credentials.

## Client

::: src.db.client

## Loader

::: src.db.loader

## Session names

::: src.db.database_utils
