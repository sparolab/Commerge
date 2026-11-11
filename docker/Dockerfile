# Commerge with every solver and the registration: ghcr.io/sparolab/commerge
#   docker run --rm -v $HOME/robot_a:/a -v $HOME/robot_b:/b -v $PWD/merged:/out ghcr.io/sparolab/commerge \
#       commerge /a /b --threshold 0.96 --min-area 2000 --out /out --plot
# Gurobi: mount your license, -v $HOME/gurobi.lic:/opt/gurobi/gurobi.lic (otherwise HiGHS is used).
FROM python:3.12-slim
ENV GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic PYTHONUNBUFFERED=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/commerge
COPY pyproject.toml README.md LICENSE ./
COPY python python
RUN pip install --no-cache-dir ".[gurobi,solid,compare]" && commerge --version
WORKDIR /work
CMD ["commerge", "--help"]
