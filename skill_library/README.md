# Skill Library

This directory stores the learned skills acquired by the Voyager agent.

## Structure

```
skill_library/
├── README.md          # This file
├── skills.json        # Index of all skills (generated)
├── vectordb/          # Vector database for skill retrieval
└── code/              # Skill code files
    ├── mine_wood.js
    ├── craft_pickaxe.js
    └── ...
```

## Usage

Skills are automatically saved when Voyager learns new abilities.
You can also manually add skills by creating JavaScript files in the `code/` directory.

## Skill Format

Each skill is a JavaScript function that can be called by Mineflayer:

```javascript
async function skillName(bot) {
    // Skill implementation
}
```

## Pre-built Skills

The repository includes some pre-built skills in subdirectories.
These provide a foundation for common Minecraft tasks.
