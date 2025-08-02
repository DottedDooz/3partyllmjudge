import os
import json
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("OpenAI API key not found in environment variable OPENAI_API_KEY")

data_folder = 'path\\to\\data'
jsonl_file = os.path.join(data_folder, 'llm_prompts.jsonl')
output_file = os.path.join(data_folder, 'llm_results.json')

def get_llm_response(system_prompt, user_prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=10,
        n=1,
        stop=None,
    )
    return response.choices[0].message.content.strip()

def main():
    results = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            #if i >= 3:  # limit to first 3 for testing
            #    break
            prompt_entry = json.loads(line)
            game_id = prompt_entry['game_id']
            ai_witness_id = prompt_entry['ai_witness_id']
            system_prompt = prompt_entry['system_prompt']
            user_prompt = prompt_entry['user_prompt']
            human_label = prompt_entry['human_label']

            print(f"Game ID: {game_id}")

            llm_answer = get_llm_response(system_prompt, user_prompt)

            print(f"LLM Response: {llm_answer}")
            print(f"Correct Human Label: {human_label}")
            print("---")

            results.append({
                'game_id': game_id,
                'ai_witness_id': ai_witness_id,
                'llm_response': llm_answer,
                'human_label': human_label
            })

    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"Saved results for {len(results)} games to {output_file}")

if __name__ == "__main__":
    main()
