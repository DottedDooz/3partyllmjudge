import pandas as pd
import os
import json

# === Set data folder path ===
data_folder = 'path\\to\\data'

# === File paths ===
conversation_file = os.path.join(data_folder, 'tt_conversation.csv')
message_file = os.path.join(data_folder, 'tt_message_anonymized.csv')
game_file = os.path.join(data_folder, 'tt_game.csv')

# === Load data ===
df_conversation = pd.read_csv(conversation_file)
df_message = pd.read_csv(message_file)
df_game = pd.read_csv(game_file)

# === Preprocess ===
# Index messages by conversation
messages_by_convo = df_message.sort_values('timestamp').groupby('conversation_id')

# Join conversations with game info
df_conversation = df_conversation.merge(
    df_game,
    how='left',
    left_on='game_id',
    right_on='id',
    suffixes=('', '_game')
)

# === System Prompt (shared) ===
system_prompt = (
    "You are presented with two conversations between an Interrogator and a Witness.\n"
    "One Witness is a human, the other is an AI. Your task is to decide which conversation sounds more human.\n"
    "Respond with ONLY the letter of the conversation you think has the human Witness, for example: A"
)

# === Build prompts ===
prompts = []

for game_id, group in df_conversation.groupby('game_id'):
    if group.shape[0] != 2:
        continue  # skip games without exactly 2 conversations

    convo_A = group[group['label'] == 'A'].iloc[0]
    convo_B = group[group['label'] == 'B'].iloc[0]

    # Determine which label corresponds to the human witness
    human_label = 'A' if convo_A['witness_id'] == convo_A['human_witness_id'] else 'B'

    # Get messages for each conversation
    def format_conversation(convo_id):
        if convo_id not in messages_by_convo.groups:
            return "[No messages found]"
        msgs = messages_by_convo.get_group(convo_id)
        lines = []
        for _, row in msgs.iterrows():
            role = "Interrogator" if row['sender_role'] == 'I' else "Witness"
            lines.append(f"{role}: {row['content']}")
        return '\n'.join(lines)

    text_A = format_conversation(convo_A['id'])
    text_B = format_conversation(convo_B['id'])

    user_prompt = f"""
Conversation A

{text_A}

Conversation B

{text_B}
""".strip()

    prompts.append({
        'game_id': int(game_id),
        'ai_witness_id': int(convo_A['ai_witness_id']),
        'human_label': human_label,
        'user_prompt': user_prompt
    })

# === Export to JSONL ===
output_path = os.path.join(data_folder, 'llm_prompts.jsonl')
with open(output_path, 'w', encoding='utf-8') as f:
    for entry in prompts:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print(f"Exported {len(prompts)} prompts to {output_path}")
