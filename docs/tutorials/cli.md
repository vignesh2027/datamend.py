# CLI Guide

datamend ships a full command-line interface. No Python required.

## repair

```bash
datamend repair data.csv
datamend repair data.csv -o clean_data.csv
datamend repair data.csv --strategy median --html dashboard.html --report report.json
datamend repair data.csv --fast          # fast mode for large files
datamend repair data.csv --confirm       # ask for confirmation before applying
```

## contract

```bash
datamend contract training.csv -o my_contract.json
datamend contract training.csv --name "production_v1" --null-threshold 0.02
```

## validate

```bash
datamend validate prod.csv my_contract.json
datamend validate prod.csv contract.json --fail-fast    # exit code 1 on violations
datamend validate prod.csv contract.json --report violations.json --html report.html
```

## drift

```bash
datamend drift training.csv production.csv
datamend drift train.csv prod.csv --report drift.json --html drift.html --alpha 0.01
```

## score

```bash
datamend score mydata.csv
# MendScore: 47.3/100
```

## dashboard

```bash
datamend dashboard repair_report.json
datamend dashboard drift_report.json --port 9000 --no-open
```

## plugins

```bash
datamend plugins
# Registered datamend plugins:
#   repair: my_domain_repair, uppercase_cleaner
#   validators: (none)
#   drift: (none)
#   tracers: (none)
```
