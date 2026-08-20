# Changelog

## Unreleased

- Fix `recall_fact` so returned user and project memories update and persist
  `reuseCount` and `lastUsedAt`, just like service-layer queries.
- Refresh matching cross-project `experience` items in their source project
  when they are returned.
- Add regression coverage for both retention updates.
