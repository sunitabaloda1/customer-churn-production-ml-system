# Incident Response Plan

## Scenario

The incoming customer dataset contains unexpected schema changes or missing mandatory columns.

## Impact

- Data ingestion fails.
- Model prediction requests may return errors.
- Downstream preprocessing cannot continue.

## Detection

- Data quality validation detects missing columns or abnormal data.
- Application logs capture the error.
- Monitoring alerts notify the support team.

## Response

1. Stop the ingestion process.
2. Review application logs.
3. Validate the input dataset schema.
4. Restore the expected dataset format.
5. Restart the ingestion pipeline.
6. Verify successful preprocessing and prediction.

## Recovery

- Reload the corrected dataset.
- Execute the preprocessing pipeline.
- Validate API functionality.
- Resume normal operation.