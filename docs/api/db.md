# Database

The `src.db` package provides a MongoDB client with optional SSH tunnel support and loaders for the raw neural data.

!!! note "Credentials required"
    All functions in this module require a `.env` file with `MONGODB_URI` and `MONGODB_DATABASE` set. The CI tests need neither, they mock the client.

## Client

::: src.db.client

## Loader

::: src.db.loader

## Utilities

::: src.db.database_utils
