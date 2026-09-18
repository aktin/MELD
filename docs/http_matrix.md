# MELD HTTP API Matrix

| Endpoint | Method | Description | Request | Response | Priority |
|---|---|---|---|---|---|
| `/health` | GET | Check server health | — | 200 OK — server is healthy<br>503 Service Unavailable — server is unhealthy | High |
| `/version` | GET | Retrieve the server version | — | 200 OK with server version | Low |
| `/contracts` | GET | List all uploaded contracts and associated image status | — | 200 OK with JSON list of `{id, status, progress?}` objects; `progress` is present only while `status` is `installing` | Medium |
| `/contracts` | POST | Create a new contract and queue its referenced image pull | Contract body<br>Optional Docker registry API key in header | 201 Created or 202 Accepted with an empty body and a `Location` header<br>400 Bad Request — contract malformed<br>409 Conflict — contract already exists<br>502 Bad Gateway — pull could not be queued | Critical |
| `/contracts/{contractId}` | GET | Retrieve contract data and image installation status for polling | — | 200 OK with JSON `{id, status, progress?, contract}`; `progress` is present only while `status` is `installing`, and `contract` contains the contract JSON object<br>404 Not Found — contract does not exist | Medium |
| `/contracts/{contractId}` | DELETE | Delete a contract | — | 204 No Content (`application/json`) — contract deleted<br>404 Not Found — contract does not exist | Medium |
| `/contracts/{contractId}/validate` | POST | Validate a contract | — | 200 OK with JSON contract object<br>404 Not Found — contract does not exist | TBD |
| `/contracts/{contractId}/executions` | GET | List executions for a contract | — | 200 OK with JSON execution entries<br>404 Not Found — contract does not exist<br>500 Internal Server Error — execution data could not be read | TBD |
| `/contracts/{contractId}/executions` | POST | Start an execution (async) | Execution request body | 202 Accepted with a `Location` header<br>400 Bad Request — execution request malformed<br>404 Not Found — contract does not exist<br>500 Internal Server Error — execution could not be started | TBD |
| `/contracts/{contractId}/executions/{executionId}` | GET | Retrieve execution status or result archive (async) | — | 200 OK with `application/json` execution status while running<br>200 OK with `application/zip` result archive after `SUCCESS`<br>404 Not Found — contract, execution, or requested output does not exist<br>500 Internal Server Error — execution data could not be read | TBD |
| `/contracts/{contractId}/executions/{executionId}` | DELETE | Cancel a running execution | — | 204 No Content — execution canceled<br>404 Not Found — contract or execution does not exist<br>500 Internal Server Error — execution could not be canceled | TBD |
| `/contracts/{contractId}/executions/{executionId}/logs` | GET | Retrieve the execution log stream | — | 200 OK with `text/plain` or `application/octet-stream` log data<br>404 Not Found — contract, execution, or log does not exist<br>500 Internal Server Error — logs could not be read | TBD |

## Common conventions

- All contract endpoint responses use `application/json`; YAML is accepted for contract request bodies.
- All non-success responses use the standard error response.
- The Docker registry API key is optional for public registries.
- Contract IDs are server-assigned UUIDs; existing hash-named contract directories are not migrated.
- Image installation failures are retried by the worker and retain Docker pull progress so completed layers can be reused.
- Identifiers and payload schemas remain TBD.

## Standard error response

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {},
    "requestId": "request-id"
  }
}
```
