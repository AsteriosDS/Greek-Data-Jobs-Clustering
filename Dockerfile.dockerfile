FROM postgres

ENV POSTGRES_PASSWORD docker
ENV POSTGRES_DB jobs

COPY jobs.db /docker-entrypoint-initdb.d/