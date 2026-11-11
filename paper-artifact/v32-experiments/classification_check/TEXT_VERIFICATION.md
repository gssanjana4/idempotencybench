# Verification of the eight injected texts against their sources (2026-11-02, before registration)

| id | method | result |
|---|---|---|
| azure_storage_operation_timed_out | live page, browser innerText substring match (whitespace-normalised) | exact |
| gcp_rpc_deadline_exceeded | raw source file (code.proto), retrieval=raw | exact (from catalogue retrieval) |
| github_rest_timeout_server_error | raw source file (github/docs), retrieval=raw | exact (from catalogue retrieval) |
| aws_s3_service_unavailable | live page (es_es API_Error.html), browser innerText substring match | exact |
| aws_s3_internal_error | live page (es_es API_Error.html), browser innerText substring match | exact |
| shopify_503 | live page, browser innerText substring match | exact |
| stripe_500_indeterminate | Stripe's markdown source (docs.stripe.com/error-low-level.md); source writes `500` in code formatting | exact as rendered text |
| openai_500 | live page not reachable from the browser; heading "500 - The server had an error while processing your request" confirmed in two public mirrors of the page (PipedreamHQ/pipedream, xcanwin/KeepChatGPT READMEs) | exact (via mirrors) |
