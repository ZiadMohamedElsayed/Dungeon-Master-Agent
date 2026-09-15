# Lore

Sample world lore for the Dungeon Master Agent. Upload these files through the
frontend (**World Lore** sidebar → drop a scroll) or via the API:

```bash
curl -X POST http://localhost:8000/api/documents/lore/upload \
  -F "file=@lore/emberfall_lore.md"
```

Supported formats: `.txt`, `.pdf`, `.md` (max 10 MB per file).

## Files

- **`emberfall_lore.md`** — A complete starter setting: the village of Emberfall
  beneath the Shadow Peaks (locations, 3 factions, 4 NPCs, magic rules,
  3 adventure hooks). Already verified to ground the DM's narration.

## Adding your own

1. Drop a `.md`/`.txt`/`.pdf` file in this folder.
2. Upload it through the UI or the API call above.
3. Ask the DM about your world — answers cite the file as `sources.lore`.
4. Remove a file from the active session with the ✕ button next to it in the
   sidebar (the repo copy is kept).
