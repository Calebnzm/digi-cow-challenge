
import json
import os

nb_path = "/home/nzioka/Desktop/CS/Zindi/DigiCow Farmer Training Adoption Challenge/random_forest.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Fix combined_df assignment
# Find cell with "combined_df = pd.concat"
# Original source lines:
# "combined_df = pd.concat([df, prior_df], ignore_index=True)\n",
# "print(combined_df.info())"

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "combined_df = pd.concat" in source:
            print("Found combined_df cell.")
            # Update source to assign to df
            new_source = [
                "combined_df = pd.concat([df, prior_df], ignore_index=True)\n",
                "df = combined_df.copy()\n",
                "print(df.info())"
            ]
            cell['source'] = new_source
            break

# 2. Fix topic parsing
# Find cell with "clean_topic"
# Original source starts with: "# Create topic features"

new_topic_logic = [
    "# Create topic features\n",
    "# The 'topics_list' column contains string representations of lists (sometimes nested)\n",
    "# We need to safely parse these strings, flatten any nested lists, and clean the topics\n",
    "\n",
    "import ast\n",
    "\n",
    "def clean_and_flat_topics(topic_str):\n",
    "    if not isinstance(topic_str, str) or pd.isna(topic_str):\n",
    "        return []\n",
    "    \n",
    "    try:\n",
    "        # safely evaluate the string literal\n",
    "        parsed = ast.literal_eval(topic_str)\n",
    "    except (ValueError, SyntaxError):\n",
    "        return []\n",
    "\n",
    "    # Flatten logic\n",
    "    flat_topics = []\n",
    "    \n",
    "    def flatten(item):\n",
    "        if isinstance(item, list):\n",
    "            for sub in item:\n",
    "                flatten(sub)\n",
    "        elif isinstance(item, str):\n",
    "            flat_topics.append(item)\n",
    "            \n",
    "    flatten(parsed)\n",
    "    \n",
    "    # Cleaning: lowercase, strip, unique\n",
    "    cleaned = sorted(list(set([t.lower().strip() for t in flat_topics if t])))\n",
    "    return cleaned\n",
    "\n",
    "# Parse and clean topics from topics_list\n",
    "df['clean_topics'] = df['topics_list'].apply(clean_and_flat_topics)\n",
    "\n",
    "# Show some examples\n",
    "print(\"Sample parsed topics:\")\n",
    "print(df[['topics_list', 'clean_topics']].head())\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def clean_topic" in source and "parse_topics" in source:
            print("Found clean_topic cell.")
            cell['source'] = new_topic_logic
            break

# Save back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4) # indent=1 to match original roughly, or 4 is standard
    
print("Notebook updated successfully.")
