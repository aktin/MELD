# MELD HTTP API Matrix

| Endpoint | Method | Description | Request | Response | Priority |
|---|---|---|---|---|---|
| `/health` | GET | Check server health | — | 200 OK — server is healthy<br>503 Service Unavailable — server is unhealthy | High |
| `/version` | GET | Retrieve the server version | — | 200 OK with server version | Low |
| `/contracts` | GET | List all uploaded contracts and associated image status | — | 200 OK with JSON list of contracts | Medium |
| `/contracts` | POST | Create a new contract and queue its referenced image pull | Contract body<br>Optional Docker registry API key in header | 201/202 with JSON contract info (`id`, `contract`, `contract_json`, `pull_status`)<br>400 Bad Request — contract malformed<br>409 Conflict — contract already exists<br>502 Bad Gateway — pull could not be queued | Critical |
| `/contracts/{contractId}` | GET | Retrieve contract data and pull status for polling | — | 200 OK with JSON `{id: <sha256>, contract: <yaml string>, contract_json: <contract object>, pull_status: <status object>}`<br>404 Not Found — contract does not exist | Medium |
| `/contracts/{contractId}` | DELETE | Delete a contract | — | 204 No Content (`application/json`) — contract deleted<br>404 Not Found — contract does not exist | Medium |
| `/contracts/{contractId}/validate` | POST | Validate a contract | — | 200 OK with JSON contract object<br>404 Not Found — contract does not exist | TBD |
| `/contracts/{contractId}/inferences` | GET | List inferences for a contract | — | 200 OK with list of inferences<br>404 Not Found — contract does not exist | TBD |
| `/contracts/{contractId}/inferences` | POST | Start an inference (async) | Inference request body | 202 Accepted — inference started with inference ID<br>400 Bad Request — inference request malformed<br>404 Not Found — contract does not exist | TBD |
| `/contracts/{contractId}/inferences/{inferenceId}` | GET | Retrieve inference status or result archive (async) | — | 200 OK with inference status if still running<br>200 OK with result archive as a file object if completed<br>404 Not Found — contract or inference does not exist | TBD |
| `/contracts/{contractId}/inferences/{inferenceId}` | DELETE | Cancel a running inference | — | 204 No Content — inference canceled<br>404 Not Found — contract or inference does not exist | TBD |
| `/contracts/{contractId}/inferences/{inferenceId}/logs` | GET | Retrieve the inference log stream | — | 200 OK with log stream<br>404 Not Found — contract or inference does not exist | TBD |
| `/contracts/{contractId}/schedules` | GET | List schedules for a contract | — | 200 OK with list of schedules<br>404 Not Found — contract does not exist | TBD |
| `/contracts/{contractId}/schedules` | POST | Create a schedule for a contract | Schedule body | 201 Created — schedule created<br>400 Bad Request — schedule malformed or invalid<br>404 Not Found — contract does not exist | TBD |
| `/contracts/{contractId}/schedules/{scheduleId}` | GET | Retrieve a schedule | — | 200 OK with schedule information<br>404 Not Found — contract or schedule does not exist | TBD |
| `/contracts/{contractId}/schedules/{scheduleId}` | DELETE | Delete a schedule | — | 204 No Content — schedule deleted<br>404 Not Found — contract or schedule does not exist | TBD |

## Common conventions

- All contract endpoint responses use `application/json`; YAML is accepted for contract request bodies.
- All non-success responses use the standard error response.
- The Docker registry API key is optional for public registries.
- Pull authentication failures are reported through the `pull_status` object returned by the contract GET endpoint.
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
