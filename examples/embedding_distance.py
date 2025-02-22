import numpy as np
import ollama
import matplotlib.pyplot as plt

def cosine_distance(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def softmax(x, temperature=0.2):
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    x = np.array(x)
    exp_x = np.exp(x/temperature)
    return exp_x / np.sum(exp_x)

input = '''Ellie moved gracefully through the community garden, noticing a young woman struggling with several heavy bags of soil. Without hesitation, she offered her assistance, smiling warmly as she introduced herself and insisted on carrying the heavier load despite the woman's gentle protests. As they worked together, Ellie listened attentively to the stranger's innovative ideas for vertical gardening using recycled materials, her eyes lighting up with genuine fascination. Later, when another gardener expressed frustration about the placement of certain plants, Ellie patiently mediated the disagreement, finding a compromise that satisfied everyone while incorporating elements from various global gardening traditions she'd studied during her travels. At the garden's planning meeting that evening, she enthusiastically supported each person's vision, building upon their ideas with thoughtful suggestions drawn from her eclectic knowledge of permaculture, indigenous farming practices, and contemporary sustainable agriculture—all while making sure the quieter members had opportunities to share their perspectives and ensuring that everyone felt their contributions were valued.'''

similarity_array = []

personality_traits = {
    "openness": ["curious", "creative", "imaginative", "artistic", "insightful", "original", "unconventional", "innovative", "reflective", "adventurous"],
    
    "conscientiousness": ["organized", "responsible", "disciplined", "efficient", "thorough", "reliable", "precise", "methodical", "deliberate", "persevering"],
    
    "extraversion": ["sociable", "energetic", "talkative", "assertive", "enthusiastic", "outgoing", "gregarious", "expressive", "lively", "confident"],
    
    "agreeableness": ["kind", "cooperative", "compassionate", "considerate", "empathetic", "generous", "trusting", "forgiving", "patient", "modest"],
    
    "neuroticism": ["anxious", "moody", "irritable", "vulnerable", "self-conscious", "sensitive", "tense", "restless", "insecure", "worrying"]
}

response_input = ollama.embed(
    model="nomic-embed-text",
    input=input
)

# print("Embedding:", response_input)

input_embedding = np.array(response_input['embeddings'][0])


for key in personality_traits:
    response_personality = ollama.embed(
        model="nomic-embed-text",
        input=personality_traits[key]
    )
    personality_embedding = np.array(response_personality['embeddings'][0])
    similarity_score = cosine_distance(input_embedding, personality_embedding)
    similarity_array.append(similarity_score)


openness_array = []
conscientiousness_array = []
extraversion_array = []
agreeableness_array = []
neuroticism_array = []

temperatures = np.arange(0.05, 1.05, 0.05)

for temp in temperatures:
    openness_array.append(softmax(similarity_array, temp)[0])
    conscientiousness_array.append(softmax(similarity_array, temp)[1])
    extraversion_array.append(softmax(similarity_array, temp)[2])
    agreeableness_array.append(softmax(similarity_array, temp)[3])
    neuroticism_array.append(softmax(similarity_array, temp)[4])

plt.figure(figsize=(12, 9))
plt.plot(temperatures, openness_array, label='Openness', marker='o')
plt.plot(temperatures, conscientiousness_array, label='Conscientiousness', marker='s')
plt.plot(temperatures, extraversion_array, label='Extraversion', marker='^')
plt.plot(temperatures, agreeableness_array, label='Agreeableness', marker='d')
plt.plot(temperatures, neuroticism_array, label='Neuroticism', marker='v')

plt.xticks(np.arange(0, 1.05, 0.05))
plt.yticks(np.arange(0, 0.75, 0.05))
plt.xlabel('Temperature')
plt.ylabel('Softmax Probability')
plt.title('Personality Traits Probabilities vs Temperature')
plt.legend()
plt.grid(True)
plt.show()