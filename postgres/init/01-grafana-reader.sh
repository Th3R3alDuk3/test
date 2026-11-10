#!/bin/sh
# Read-only role for Grafana. agentgateway creates its tables (request_logs, ...)
# on first start, so SELECT is granted through default privileges.
set -eu

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v reader="$GRAFANA_DB_USER" -v reader_password="$GRAFANA_DB_PASSWORD" \
  -v owner="$POSTGRES_USER" -v db="$POSTGRES_DB" <<'SQL'
CREATE ROLE :"reader" LOGIN PASSWORD :'reader_password';
GRANT CONNECT ON DATABASE :"db" TO :"reader";
GRANT USAGE ON SCHEMA public TO :"reader";
GRANT SELECT ON ALL TABLES IN SCHEMA public TO :"reader";
ALTER DEFAULT PRIVILEGES FOR ROLE :"owner" IN SCHEMA public GRANT SELECT ON TABLES TO :"reader";
SQL
