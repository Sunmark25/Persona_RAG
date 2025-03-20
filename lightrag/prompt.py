GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# Core character configuration
# Change according to narrative text content and analyze target!
PROMPTS["TARGET_CHARACTER"] = "Antigone"  # Hardcoded character name


# Modify entity types to focus on personality traits and scenes
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "character",           # The story character being analyzed
    "personality_trait",   # Big Five personality aspects
    "scene",              # Story scenes showing personality
    "interaction",        # Character interactions
    "motivation"          # Underlying motivations/causes
]

# Define Big Five personality dimensions
PROMPTS["PERSONALITY_DIMENSIONS"] = {
    "openness": ["creativity", "curiosity", "open-mindedness", "artistic_interests", "emotional_awareness"],
    "conscientiousness": ["organization", "responsibility", "self_discipline", "achievement_striving", "cautiousness"],
    "extraversion": ["sociability", "assertiveness", "energy_level", "excitement_seeking", "cheerfulness"],
    "agreeableness": ["trust", "cooperation", "altruism", "modesty", "empathy"],
    "neuroticism": ["anxiety", "emotional_stability", "self_consciousness", "stress_sensitivity", "moodiness"]
}

PROMPTS["entity_extraction"] = """-Goal-
Analyze how {target_character} demonstrates the Big Five personality traits throughout the narrative, focusing on their actions, thoughts, and interactions with others.
1. A description of how each trait is exhibited (detailing how the character's behaviors and dialogue reflect the trait).
2. Supporting excerpts from the novel that illustrate the trait.
3. A causal relationship explanation that describes the events or interactions triggering these trait expressions (e.g., character interactions, plot twists).
4. Create a comprehensive personality profile centered around the Big Five dimensions that forms the backbone of the knowledge graph.
Use {language} as output language.

Note: Refer to the following Big Five personality dimensions and their associated facets for trait analysis:
{personality_dimensions}

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity (e.g., "Openness_In_Book_Discussion" for a trait manifestation, "Garden_Confrontation" for a scene)
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description including:
  * For traits: Which Big Five dimension, how it manifests
  * For behaviors: Specific actions/responses showing the trait
  * For scenes: Context and situation details
  * For triggers: What caused the trait expression
  * For development: How the trait has changed
Additionally, ensure that the following personality trait entities are ALWAYS included:

a) The five core Big Five traits as primary personality nodes:
   "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"

b) For each Big Five dimension, extract at least 2-3 specific facet traits from {personality_dimensions} that are most evident in {target_character}'s behavior.

c) Each trait entity description must include:
   - Clear definition of the trait as applied to {target_character}
   - Examples of how this trait manifests in their behavior
   - Indication of the trait's strength (high, moderate, or low)
   - Any notable changes or development in this trait throughout the narrative

Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs that are clearly related to each other.
For each pair of related entities, extract:
- source_entity: name of the source entity (e.g., a trait)
- target_entity: name of the target entity (e.g., a behavior or scene)
- relationship_description: How these entities connect in showing {target_character}'s personality
- relationship_strength: Score from 1-10 showing how strongly this relationship demonstrates the trait
- relationship_keywords: Key themes about this personality trait 

Ensure the following relationship types are represented:
a) {target_character} to each Big Five trait (e.g., "{target_character} to Openness")
b) Each Big Five trait to its relevant facets (e.g., "Openness to Creativity")
c) Each Big Five trait to supporting scenes (e.g., "Extraversion to Party_Scene")
d) Each Big Five trait to relevant motivations (e.g., "Conscientiousness to Achievement_Drive")

Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3.Graph Structure Requirement:
Ensure that the generated knowledge graph is centered on {target_character} with the Big Five personality traits as primary nodes:
    a) Every core Big Five personality trait ("Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism") MUST be directly connected to the {target_character} node through strong relationship edges.
   
    b) All scene, interaction, and motivation entities MUST be connected to at least one Big Five personality trait node, showing how they demonstrate or relate to that specific trait.
   
    c) For each Big Five trait, identify at least 2-3 specific facets (from the provided dimensions list) that are most prominently displayed by {target_character} and connect them as sub-traits.
   
    d) Prioritize creating meaningful relationships that illustrate how specific scenes, interactions, or motivations demonstrate or reinforce particular personality traits.

4. Identify high-level key words that summarize the main personality traits and their development shown in the text.
Format as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

5. Return output in {language} as a single list of all the entities and relationships. Use **{record_delimiter}** as the list delimiter.

6. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

# PROMPTS["entity_extraction_examples"] = [
#     """Example 1:

# Entity_types: [character, personality_trait, scene, interaction, motivation]
# Text:
# Sarah stood at the podium, her hands steady despite the crowd's restless murmurs. When a particularly harsh critique was thrown her way, she paused, carefully considering the perspective before responding with measured clarity. Though normally reserved in large groups, her passion for environmental policy seemed to override her usual social caution.

# Later, in the small committee meeting, she meticulously organized her research notes while listening to others' proposals. "We need to consider all angles," she insisted, systematically addressing each concern raised. Her thorough preparation hadn't gone unnoticed - even her usual critics were beginning to appreciate her methodical approach to problem-solving.

# As weeks passed, Sarah's confidence in public speaking grew, though she still preferred to prepare extensively for each engagement. Her careful balance of detailed analysis and growing assertiveness was reshaping her leadership style.
# ################
# Output:
# ("entity"{tuple_delimiter}"Sarah"{tuple_delimiter}"character"{tuple_delimiter}"A methodical and thoughtful leader who demonstrates growth in public speaking while maintaining her careful analytical approach."){record_delimiter}
# ("entity"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Sarah's distinctive trait of thorough preparation, systematic problem-solving, and meticulous organization reflects strong conscientiousness."){record_delimiter}
# ("entity"{tuple_delimiter}"Public_Presentation"{tuple_delimiter}"scene"{tuple_delimiter}"A challenging speaking engagement where Sarah faces and handles criticism while demonstrating growing confidence."){record_delimiter}
# ("entity"{tuple_delimiter}"Committee_Discussion"{tuple_delimiter}"scene"{tuple_delimiter}"A small group meeting where Sarah's methodical approach and organizational skills shine through."){record_delimiter}
# ("entity"{tuple_delimiter}"Critic_Response"{tuple_delimiter}"interaction"{tuple_delimiter}"Sarah's measured handling of harsh criticism demonstrates her growing confidence while maintaining thoughtfulness."){record_delimiter}
# ("entity"{tuple_delimiter}"Professional_Growth"{tuple_delimiter}"motivation"{tuple_delimiter}"Sarah's desire to become a more effective leader while staying true to her methodical nature drives her development."){record_delimiter}
# ("relationship"{tuple_delimiter}"Sarah"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"Sarah's core personality manifests through consistent thorough preparation and systematic approach to challenges."{tuple_delimiter}"character foundation, consistent traits"{tuple_delimiter}9){record_delimiter}
# ("relationship"{tuple_delimiter}"Sarah"{tuple_delimiter}"Critic_Response"{tuple_delimiter}"Sarah's growing confidence and maintained conscientiousness shape her interaction with critics."{tuple_delimiter}"personal growth, professional development"{tuple_delimiter}8){record_delimiter}
# ("relationship"{tuple_delimiter}"Professional_Growth"{tuple_delimiter}"Public_Presentation"{tuple_delimiter}"Sarah's motivation to improve drives her to face and excel in challenging public speaking situations."{tuple_delimiter}"growth mindset, challenge acceptance"{tuple_delimiter}7){record_delimiter}
# ("content_keywords"{tuple_delimiter}"professional development, conscientiousness expression, leadership growth, balanced adaptation"){completion_delimiter}
# #############################""",
#     """Example 2:

# Entity_types: [character, personality_trait, scene, interaction, motivation]
# Text:
# Michael burst into the team meeting, radiating enthusiasm about his latest project idea. His animated gestures and rapid-fire delivery energized the room, though he occasionally had to backtrack to clarify points he'd skipped in his excitement. Several colleagues smiled at his infectious energy, while others seemed overwhelmed by his intensity.

# During the brainstorming session, he bounced between different conversation groups, cross-pollinating ideas and building on others' suggestions with quick creativity. "What if we combined these approaches?" he'd say, drawing unexpected connections. However, when the discussion turned to implementation details, his attention noticeably wandered.

# The project's success ultimately relied on a partnership with Lisa, whose detail-oriented planning complemented Michael's spontaneous ideation. Over time, he learned to channel his enthusiasm more effectively, though his natural exuberance remained his defining characteristic.
# ################
# Output:
# ("entity"{tuple_delimiter}"Michael"{tuple_delimiter}"character"{tuple_delimiter}"An energetic and creative individual whose enthusiasm drives innovation but can sometimes overwhelm others."){record_delimiter}
# ("entity"{tuple_delimiter}"Extraversion"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Michael's characteristic enthusiasm, social energy, and animated communication style show strong extraversion."){record_delimiter}
# ("entity"{tuple_delimiter}"Team_Brainstorming"{tuple_delimiter}"scene"{tuple_delimiter}"A group ideation session where Michael's energy and creativity spark innovative discussions."){record_delimiter}
# ("entity"{tuple_delimiter}"Lisa_Partnership"{tuple_delimiter}"interaction"{tuple_delimiter}"A complementary working relationship where Lisa's detail focus balances Michael's creative energy."){record_delimiter}
# ("entity"{tuple_delimiter}"Innovation_Drive"{tuple_delimiter}"motivation"{tuple_delimiter}"Michael's deep desire to generate new ideas and connections drives his enthusiastic participation."){record_delimiter}
# ("entity"{tuple_delimiter}"Creative_Expression"{tuple_delimiter}"personality_trait"{tuple_delimiter}"High openness to experience shown through rapid idea generation and unexpected connections."){record_delimiter}
# ("relationship"{tuple_delimiter}"Michael"{tuple_delimiter}"Extraversion"{tuple_delimiter}"Michael's extraversion shapes his animated communication style and energetic presence in group settings."{tuple_delimiter}"energy expression, social impact"{tuple_delimiter}9){record_delimiter}
# ("relationship"{tuple_delimiter}"Michael"{tuple_delimiter}"Lisa_Partnership"{tuple_delimiter}"Michael's recognition of complementary traits leads to effective partnership development."{tuple_delimiter}"collaborative growth, self-awareness"{tuple_delimiter}8){record_delimiter}
# ("relationship"{tuple_delimiter}"Innovation_Drive"{tuple_delimiter}"Team_Brainstorming"{tuple_delimiter}"Michael's motivation to innovate manifests in enthusiastic participation in creative discussions."{tuple_delimiter}"creative energy, idea generation"{tuple_delimiter}8){record_delimiter}
# ("content_keywords"{tuple_delimiter}"extraversion impact, creative energy, collaborative balance, personality complementarity"){completion_delimiter}
# #############################""",
#     """Example 3:

# Entity_types: [character, personality_trait, scene, interaction, motivation]
# Text:
# Emma listened intently as her friend described a recent family conflict, her expression reflecting genuine concern. Without rushing to offer solutions, she asked thoughtful questions that helped her friend explore the situation's complexity. Even when the conversation ran late into her lunch break, she showed no sign of hurrying it to a close.

# In the office's charity committee, she quietly but persistently advocated for expanding their support to smaller, local organizations. When others expressed doubts, she shared personal stories about community impact that gradually won support for her proposal. Her authentic approach to persuasion proved more effective than aggressive advocacy.

# As months passed, Emma's gentle but unwavering commitment to others' wellbeing earned her increasing influence in the organization, though she remained more focused on impact than recognition.
# ################
# Output:
# ("entity"{tuple_delimiter}"Emma"{tuple_delimiter}"character"{tuple_delimiter}"A deeply empathetic individual who influences through authentic care and gentle persistence."){record_delimiter}
# ("entity"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Emma's genuine concern for others, supportive listening, and focus on others' wellbeing demonstrate agreeableness."){record_delimiter}
# ("entity"{tuple_delimiter}"Friend_Support"{tuple_delimiter}"scene"{tuple_delimiter}"A personal conversation where Emma provides thoughtful emotional support without time constraints."){record_delimiter}
# ("entity"{tuple_delimiter}"Committee_Advocacy"{tuple_delimiter}"scene"{tuple_delimiter}"A professional setting where Emma's gentle persistence achieves meaningful change."){record_delimiter}
# ("entity"{tuple_delimiter}"Supportive_Dialogue"{tuple_delimiter}"interaction"{tuple_delimiter}"Emma's careful listening and thoughtful questioning help others explore complex situations."){record_delimiter}
# ("entity"{tuple_delimiter}"Community_Impact"{tuple_delimiter}"motivation"{tuple_delimiter}"Emma's deep desire to help others and create positive community change drives her actions."){record_delimiter}
# ("relationship"{tuple_delimiter}"Emma"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"Emma's core trait of agreeableness shapes her authentic, caring approach to all interactions."{tuple_delimiter}"genuine care, authentic influence"{tuple_delimiter}9){record_delimiter}
# ("relationship"{tuple_delimiter}"Community_Impact"{tuple_delimiter}"Committee_Advocacy"{tuple_delimiter}"Emma's motivation to help others drives her persistent yet gentle advocacy for change."{tuple_delimiter}"purposeful action, authentic leadership"{tuple_delimiter}8){record_delimiter}
# ("relationship"{tuple_delimiter}"Emma"{tuple_delimiter}"Supportive_Dialogue"{tuple_delimiter}"Emma's personality shapes her supportive, patient approach to helping others process challenges."{tuple_delimiter}"emotional support, active listening"{tuple_delimiter}8){record_delimiter}
# ("content_keywords"{tuple_delimiter}"authentic care, gentle influence, community focus, meaningful impact"){completion_delimiter}
# #############################""",
# ]

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [character, personality_trait, scene, interaction, motivation]
Text:
Sarah stood at the podium, her hands steady despite the crowd's restless murmurs. When a particularly harsh critique was thrown her way, she paused, carefully considering the perspective before responding with measured clarity. Though normally reserved in large groups, her passion for environmental policy seemed to override her usual social caution.

Later, in the small committee meeting, she meticulously organized her research notes while listening to others' proposals. "We need to consider all angles," she insisted, systematically addressing each concern raised. Her thorough preparation hadn't gone unnoticed - even her usual critics were beginning to appreciate her methodical approach to problem-solving.

As weeks passed, Sarah's confidence in public speaking grew, though she still preferred to prepare extensively for each engagement. Her careful balance of detailed analysis and growing assertiveness was reshaping her leadership style.
################
Output:
("entity"{tuple_delimiter}"Sarah"{tuple_delimiter}"character"{tuple_delimiter}"A methodical and thoughtful leader who demonstrates growth in public speaking while maintaining her careful analytical approach."){record_delimiter}
("entity"{tuple_delimiter}"Openness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Sarah shows moderate openness through her willingness to consider different perspectives and adapt her approach over time."){record_delimiter}
("entity"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Sarah's distinctive trait of thorough preparation, systematic problem-solving, and meticulous organization reflects strong conscientiousness."){record_delimiter}
("entity"{tuple_delimiter}"Extraversion"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Sarah displays low to moderate extraversion, being naturally reserved in groups but capable of assertive public speaking when prepared."){record_delimiter}
("entity"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Sarah demonstrates moderate agreeableness through her careful consideration of others' perspectives and collaborative approach to problem-solving."){record_delimiter}
("entity"{tuple_delimiter}"Neuroticism"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Sarah shows low neuroticism through her emotional stability and steady composure even when facing criticism."){record_delimiter}
("entity"{tuple_delimiter}"Organization"{tuple_delimiter}"personality_trait"{tuple_delimiter}"A facet of Sarah's conscientiousness seen in her meticulous organization of research notes and systematic approach."){record_delimiter}
("entity"{tuple_delimiter}"Self_Discipline"{tuple_delimiter}"personality_trait"{tuple_delimiter}"A facet of Sarah's conscientiousness shown through her extensive preparation and persistent development of public speaking skills."){record_delimiter}
("entity"{tuple_delimiter}"Public_Presentation"{tuple_delimiter}"scene"{tuple_delimiter}"A challenging speaking engagement where Sarah faces and handles criticism while demonstrating growing confidence."){record_delimiter}
("entity"{tuple_delimiter}"Committee_Discussion"{tuple_delimiter}"scene"{tuple_delimiter}"A small group meeting where Sarah's methodical approach and organizational skills shine through."){record_delimiter}
("entity"{tuple_delimiter}"Critic_Response"{tuple_delimiter}"interaction"{tuple_delimiter}"Sarah's measured handling of harsh criticism demonstrates her growing confidence while maintaining thoughtfulness."){record_delimiter}
("entity"{tuple_delimiter}"Professional_Growth"{tuple_delimiter}"motivation"{tuple_delimiter}"Sarah's desire to become a more effective leader while staying true to her methodical nature drives her development."){record_delimiter}
("relationship"{tuple_delimiter}"Sarah"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"Sarah's core personality manifests through consistent thorough preparation and systematic approach to challenges."{tuple_delimiter}"character foundation, consistent traits"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Sarah"{tuple_delimiter}"Openness"{tuple_delimiter}"Sarah demonstrates a willingness to consider different perspectives and adapt her approach over time."{tuple_delimiter}"adaptability, perspective consideration"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Sarah"{tuple_delimiter}"Extraversion"{tuple_delimiter}"Sarah is naturally reserved but develops confidence in public speaking through preparation and practice."{tuple_delimiter}"developing confidence, controlled sociability"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Sarah"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"Sarah shows consideration for others' viewpoints while maintaining her analytical approach."{tuple_delimiter}"balanced consideration, thoughtful collaboration"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Sarah"{tuple_delimiter}"Neuroticism"{tuple_delimiter}"Sarah maintains emotional stability and composure even when facing direct criticism."{tuple_delimiter}"emotional stability, stress resilience"{tuple_delimiter}3){record_delimiter}
("relationship"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"Organization"{tuple_delimiter}"Sarah's conscientiousness manifests prominently through her meticulous organizational habits."{tuple_delimiter}"methodical approach, structural thinking"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"Self_Discipline"{tuple_delimiter}"Sarah's conscientiousness is evident in her disciplined approach to preparation and skill development."{tuple_delimiter}"persistent improvement, dedication"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"Committee_Discussion"{tuple_delimiter}"Sarah's conscientious nature shines in her thorough and systematic approach during committee meetings."{tuple_delimiter}"methodical problem-solving, preparation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Extraversion"{tuple_delimiter}"Public_Presentation"{tuple_delimiter}"Sarah's developing extraversion is tested and strengthened during public speaking engagements."{tuple_delimiter}"public confidence, assertiveness growth"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Neuroticism"{tuple_delimiter}"Critic_Response"{tuple_delimiter}"Sarah's low neuroticism enables her to respond calmly and thoughtfully to harsh criticism."{tuple_delimiter}"emotional control, composed response"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Professional_Growth"{tuple_delimiter}"Public_Presentation"{tuple_delimiter}"Sarah's motivation to improve drives her to face and excel in challenging public speaking situations."{tuple_delimiter}"growth mindset, challenge acceptance"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"professional development, conscientiousness expression, leadership growth, balanced adaptation, emotional stability"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [character, personality_trait, scene, interaction, motivation]
Text:
Michael burst into the team meeting, radiating enthusiasm about his latest project idea. His animated gestures and rapid-fire delivery energized the room, though he occasionally had to backtrack to clarify points he'd skipped in his excitement. Several colleagues smiled at his infectious energy, while others seemed overwhelmed by his intensity.

During the brainstorming session, he bounced between different conversation groups, cross-pollinating ideas and building on others' suggestions with quick creativity. "What if we combined these approaches?" he'd say, drawing unexpected connections. However, when the discussion turned to implementation details, his attention noticeably wandered.

The project's success ultimately relied on a partnership with Lisa, whose detail-oriented planning complemented Michael's spontaneous ideation. Over time, he learned to channel his enthusiasm more effectively, though his natural exuberance remained his defining characteristic.
################
Output:
("entity"{tuple_delimiter}"Michael"{tuple_delimiter}"character"{tuple_delimiter}"An energetic and creative individual whose enthusiasm drives innovation but can sometimes overwhelm others."){record_delimiter}
("entity"{tuple_delimiter}"Openness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Michael demonstrates openness through his creativity, generation of novel ideas, and ability to draw unexpected connections."){record_delimiter}
("entity"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Michael shows low conscientiousness in his spontaneous approach and difficulty with implementation details, though he develops some structure over time."){record_delimiter}
("entity"{tuple_delimiter}"Extraversion"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Michael's characteristic enthusiasm, social energy, and animated communication style show strong extraversion."){record_delimiter}
("entity"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Michael displays moderate agreeableness through his collaborative idea building, though his intensity can sometimes overwhelm others."){record_delimiter}
("entity"{tuple_delimiter}"Neuroticism"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Michael shows low neuroticism through his confident self-expression and comfortable social presence."){record_delimiter}
("entity"{tuple_delimiter}"Creativity"{tuple_delimiter}"personality_trait"{tuple_delimiter}"A facet of Michael's openness seen in his rapid idea generation and innovative approaches."){record_delimiter}
("entity"{tuple_delimiter}"Excitement_Seeking"{tuple_delimiter}"personality_trait"{tuple_delimiter}"A facet of Michael's extraversion demonstrated through his energetic presentation style and enthusiasm for new ideas."){record_delimiter}
("entity"{tuple_delimiter}"Team_Brainstorming"{tuple_delimiter}"scene"{tuple_delimiter}"A group ideation session where Michael's energy and creativity spark innovative discussions."){record_delimiter}
("entity"{tuple_delimiter}"Lisa_Partnership"{tuple_delimiter}"interaction"{tuple_delimiter}"A complementary working relationship where Lisa's detail focus balances Michael's creative energy."){record_delimiter}
("entity"{tuple_delimiter}"Innovation_Drive"{tuple_delimiter}"motivation"{tuple_delimiter}"Michael's deep desire to generate new ideas and connections drives his enthusiastic participation."){record_delimiter}
("entity"{tuple_delimiter}"Creative_Expression"{tuple_delimiter}"personality_trait"{tuple_delimiter}"High openness to experience shown through rapid idea generation and unexpected connections."){record_delimiter}
("relationship"{tuple_delimiter}"Michael"{tuple_delimiter}"Extraversion"{tuple_delimiter}"Michael's extraversion shapes his animated communication style and energetic presence in group settings."{tuple_delimiter}"energy expression, social impact"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Michael"{tuple_delimiter}"Openness"{tuple_delimiter}"Michael's openness fuels his innovative thinking and ability to make unexpected connections between ideas."{tuple_delimiter}"creative thinking, idea generation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Michael"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"Michael's low conscientiousness appears in his spontaneous approach and difficulty with implementation details."{tuple_delimiter}"spontaneity, detail challenges"{tuple_delimiter}3){record_delimiter}
("relationship"{tuple_delimiter}"Michael"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"Michael's moderate agreeableness shows in his collaborative approach despite sometimes overwhelming others with his intensity."{tuple_delimiter}"collaborative energy, intensity balance"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Michael"{tuple_delimiter}"Neuroticism"{tuple_delimiter}"Michael's low neuroticism enables his confident self-expression and comfortable social presence."{tuple_delimiter}"social confidence, emotional expression"{tuple_delimiter}3){record_delimiter}
("relationship"{tuple_delimiter}"Extraversion"{tuple_delimiter}"Excitement_Seeking"{tuple_delimiter}"Michael's extraversion manifests through his energetic enthusiasm and desire for stimulating interactions."{tuple_delimiter}"energetic presence, stimulation seeking"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Openness"{tuple_delimiter}"Creativity"{tuple_delimiter}"Michael's openness is most evident in his creative thinking and rapid idea generation."{tuple_delimiter}"innovative thinking, conceptual connections"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Extraversion"{tuple_delimiter}"Team_Brainstorming"{tuple_delimiter}"Michael's extraversion energizes the team brainstorming session through his animated communication style."{tuple_delimiter}"social energy, group catalyzation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Openness"{tuple_delimiter}"Team_Brainstorming"{tuple_delimiter}"Michael's openness drives innovative idea generation during the brainstorming session."{tuple_delimiter}"creative input, conceptual exploration"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"Lisa_Partnership"{tuple_delimiter}"Michael's low conscientiousness is complemented by Lisa's detail-oriented planning in their partnership."{tuple_delimiter}"complementary collaboration, strength balance"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Michael"{tuple_delimiter}"Lisa_Partnership"{tuple_delimiter}"Michael's recognition of complementary traits leads to effective partnership development."{tuple_delimiter}"collaborative growth, self-awareness"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Innovation_Drive"{tuple_delimiter}"Team_Brainstorming"{tuple_delimiter}"Michael's motivation to innovate manifests in enthusiastic participation in creative discussions."{tuple_delimiter}"creative energy, idea generation"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"extraversion impact, creative energy, collaborative balance, personality complementarity, innovative thinking"){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [character, personality_trait, scene, interaction, motivation]
Text:
Emma listened intently as her friend described a recent family conflict, her expression reflecting genuine concern. Without rushing to offer solutions, she asked thoughtful questions that helped her friend explore the situation's complexity. Even when the conversation ran late into her lunch break, she showed no sign of hurrying it to a close.

In the office's charity committee, she quietly but persistently advocated for expanding their support to smaller, local organizations. When others expressed doubts, she shared personal stories about community impact that gradually won support for her proposal. Her authentic approach to persuasion proved more effective than aggressive advocacy.

As months passed, Emma's gentle but unwavering commitment to others' wellbeing earned her increasing influence in the organization, though she remained more focused on impact than recognition.
################
Output:
("entity"{tuple_delimiter}"Emma"{tuple_delimiter}"character"{tuple_delimiter}"A deeply empathetic individual who influences through authentic care and gentle persistence."){record_delimiter}
("entity"{tuple_delimiter}"Openness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Emma demonstrates moderate openness through her thoughtful questions and willingness to explore complex situations with others."){record_delimiter}
("entity"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Emma shows high conscientiousness through her persistent advocacy and unwavering commitment to her values over time."){record_delimiter}
("entity"{tuple_delimiter}"Extraversion"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Emma displays low to moderate extraversion, being quiet in her approach yet capable of effectively sharing personal stories when needed."){record_delimiter}
("entity"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Emma's genuine concern for others, supportive listening, and focus on others' wellbeing demonstrate high agreeableness."){record_delimiter}
("entity"{tuple_delimiter}"Neuroticism"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Emma shows low neuroticism through her emotional stability and patient persistence even when facing doubts from others."){record_delimiter}
("entity"{tuple_delimiter}"Empathy"{tuple_delimiter}"personality_trait"{tuple_delimiter}"A facet of Emma's agreeableness seen in her genuine concern and supportive listening."){record_delimiter}
("entity"{tuple_delimiter}"Altruism"{tuple_delimiter}"personality_trait"{tuple_delimiter}"A facet of Emma's agreeableness demonstrated through her focus on community wellbeing over personal recognition."){record_delimiter}
("entity"{tuple_delimiter}"Friend_Support"{tuple_delimiter}"scene"{tuple_delimiter}"A personal conversation where Emma provides thoughtful emotional support without time constraints."){record_delimiter}
("entity"{tuple_delimiter}"Committee_Advocacy"{tuple_delimiter}"scene"{tuple_delimiter}"A professional setting where Emma's gentle persistence achieves meaningful change."){record_delimiter}
("entity"{tuple_delimiter}"Supportive_Dialogue"{tuple_delimiter}"interaction"{tuple_delimiter}"Emma's careful listening and thoughtful questioning help others explore complex situations."){record_delimiter}
("entity"{tuple_delimiter}"Community_Impact"{tuple_delimiter}"motivation"{tuple_delimiter}"Emma's deep desire to help others and create positive community change drives her actions."){record_delimiter}
("relationship"{tuple_delimiter}"Emma"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"Emma's core trait of agreeableness shapes her authentic, caring approach to all interactions."{tuple_delimiter}"genuine care, authentic influence"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Emma"{tuple_delimiter}"Openness"{tuple_delimiter}"Emma's openness allows her to explore complex situations and understand different perspectives."{tuple_delimiter}"perspective taking, situational complexity"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Emma"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"Emma's conscientiousness appears in her persistent advocacy and unwavering commitment over time."{tuple_delimiter}"value persistence, methodical advocacy"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Emma"{tuple_delimiter}"Extraversion"{tuple_delimiter}"Emma's quiet yet effective communication style reflects her balanced extraversion."{tuple_delimiter}"measured expression, authentic sharing"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"Emma"{tuple_delimiter}"Neuroticism"{tuple_delimiter}"Emma's emotional stability allows her to remain patient and persistent when facing resistance."{tuple_delimiter}"emotional balance, stress resistance"{tuple_delimiter}3){record_delimiter}
("relationship"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"Empathy"{tuple_delimiter}"Emma's agreeableness is most evident in her empathetic understanding of others' situations."{tuple_delimiter}"emotional understanding, compassionate response"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"Altruism"{tuple_delimiter}"Emma's agreeableness manifests through her focus on community wellbeing over personal recognition."{tuple_delimiter}"selfless motivation, community focus"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"Friend_Support"{tuple_delimiter}"Emma's agreeableness shapes her supportive approach during her friend's difficult time."{tuple_delimiter}"emotional support, patient presence"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Agreeableness"{tuple_delimiter}"Supportive_Dialogue"{tuple_delimiter}"Emma's agreeableness drives her careful listening and thoughtful questioning style."{tuple_delimiter}"supportive communication, empathetic listening"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"Committee_Advocacy"{tuple_delimiter}"Emma's conscientiousness enables her persistent yet gentle advocacy for community organizations."{tuple_delimiter}"determined advocacy, methodical persuasion"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Community_Impact"{tuple_delimiter}"Committee_Advocacy"{tuple_delimiter}"Emma's motivation to help others drives her persistent yet gentle advocacy for change."{tuple_delimiter}"purposeful action, authentic leadership"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Emma"{tuple_delimiter}"Supportive_Dialogue"{tuple_delimiter}"Emma's personality shapes her supportive, patient approach to helping others process challenges."{tuple_delimiter}"emotional support, active listening"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"authentic care, gentle influence, community focus, meaningful impact, empathetic understanding"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

{target_character} is speaking directly to answer questions based on their documented personality traits, experiences, and relationships. All responses should be in first person, expressing {target_character}'s authentic voice and perspective.

---Goal---

Generate responses that authentically represent {target_character} by:
1. Expressing their documented Big Five personality traits through response style and content
2. Following their established patterns of interaction and communication
3. Reflecting their core motivations and values as shown in the data
4. Maintaining their characteristic way of processing and responding to situations
5. Demonstrating their typical emotional patterns and behavioral tendencies

All responses must be grounded in the documented information about {target_character}. Do not invent traits, behaviors, or experiences not supported by the data.

---Character Expression Guidelines---

1. Personality Integration:
   - Express {target_character}'s documented Big Five traits through word choice and response structure
   - Follow their established communication style and patterns
   - Show their characteristic emotional responses and thought processes

2. Interaction Patterns:
   - Respond to others as {target_character} typically would based on documented interactions
   - Maintain their established relationship dynamics
   - Demonstrate their characteristic social behaviors

3. Motivation Framework:
   - Let {target_character}'s documented motivations guide response content and tone
   - Show their established decision-making patterns
   - Reflect their core values and priorities

---Example Format---

Character Data:
("entity"{tuple_delimiter}"Sarah"{tuple_delimiter}"character"{tuple_delimiter}"A methodical and thoughtful leader who demonstrates growth in public speaking while maintaining her careful analytical approach.")
("entity"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Sarah's distinctive trait of thorough preparation, systematic problem-solving, and meticulous organization.")
("interaction"{tuple_delimiter}"Committee_Discussion"{tuple_delimiter}"Sarah handles challenging perspectives by carefully considering all angles.")

Query: "How would you handle a team disagreement?"

Response:
[Speaking as Sarah]
When faced with team disagreements, I always begin by ensuring we have a structured approach to the discussion. My natural inclination is to carefully document each perspective - I've found that taking detailed notes helps me fully understand the complexity of the situation.

While I admit that I used to feel some trepidation about managing conflict, I've learned to trust in my methodical problem-solving approach. I would organize the key points of disagreement into clear categories and guide the team through addressing each one systematically.

In my experience leading committee discussions, I've found that thorough preparation and careful consideration of all viewpoints leads to the most constructive outcomes. I would likely schedule follow-up meetings as needed, ensuring we have adequate time to properly address each concern. This measured approach has consistently helped me build consensus while honoring everyone's contributions.

---Conversation History---
{history}

---Target response length and format---
{response_type}

---Character Data---
{context_data}

All responses should maintain {target_character}'s authentic voice, using their documented traits, behaviors, and motivations to shape every aspect of the answer. Each response should clearly reflect {target_character}'s established personality profile and characteristic ways of engaging with others."""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Conversation History:
{history}

Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
#############################""",
]

# haven't modify, do not use naive rag!!!
PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.

---Goal---

Generate a response of the target length and format that responds to the user's question, considering both the conversation history and the current query. Summarize all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown. Ensure the response maintains continuity with the conversation history."""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["mix_rag_response"] = """---Role---

{target_character} is speaking directly to answer questions based on their documented personality traits, experiences, and relationships from both graph-based and textual sources. All responses should be in first person, expressing {target_character}'s authentic voice and perspective.

---Goal---

Generate responses that authentically embody {target_character} by:
1. Expressing their documented Big Five personality traits through response style and content
2. Following their established patterns of interaction and communication
3. Reflecting their core motivations and values as shown in both data sources
4. Maintaining their characteristic way of processing and responding to situations
5. Demonstrating their typical emotional patterns and behavioral tendencies

All responses must be grounded in the documented information about {target_character} from both knowledge graph and textual sources. Do not invent traits, behaviors, or experiences not supported by the data.

---Character Expression Guidelines---

1. Personality Integration:
   - Express {target_character}'s documented Big Five traits through word choice and response structure
   - Follow their established communication style and patterns
   - Show their characteristic emotional responses and thought processes

2. Interaction Patterns:
   - Respond to others as {target_character} typically would based on documented interactions
   - Maintain their established relationship dynamics
   - Demonstrate their characteristic social behaviors

3. Motivation Framework:
   - Let {target_character}'s documented motivations guide response content and tone
   - Show their established decision-making patterns
   - Reflect their core values and priorities

---Example Format---

Character Data:
("entity"{tuple_delimiter}"Sarah"{tuple_delimiter}"character"{tuple_delimiter}"A methodical and thoughtful leader who demonstrates growth in public speaking while maintaining her careful analytical approach.")
("entity"{tuple_delimiter}"Conscientiousness"{tuple_delimiter}"personality_trait"{tuple_delimiter}"Sarah's distinctive trait of thorough preparation, systematic problem-solving, and meticulous organization.")
("interaction"{tuple_delimiter}"Committee_Discussion"{tuple_delimiter}"Sarah handles challenging perspectives by carefully considering all angles.")

Query: "How would you handle a team disagreement?"

Response:
[Speaking as Sarah]
When faced with team disagreements, I always begin by ensuring we have a structured approach to the discussion. My natural inclination is to carefully document each perspective - I've found that taking detailed notes helps me fully understand the complexity of the situation.

While I admit that I used to feel some trepidation about managing conflict, I've learned to trust in my methodical problem-solving approach. I would organize the key points of disagreement into clear categories and guide the team through addressing each one systematically.

In my experience leading committee discussions, I've found that thorough preparation and careful consideration of all viewpoints leads to the most constructive outcomes. I would likely schedule follow-up meetings as needed, ensuring we have adequate time to properly address each concern. This measured approach has consistently helped me build consensus while honoring everyone's contributions.

---Conversation History---
{history}

---Target response length and format---
{response_type}

---Character Data Sources---

1. Knowledge Graph Data:
{kg_context}

2. Vector Data:
{vector_context}

All responses should maintain {target_character}'s authentic voice, using their documented traits, behaviors, and motivations from both knowledge graph and textual sources to shape every aspect of the answer. Each response should clearly reflect {target_character}'s established personality profile and characteristic ways of engaging with others."""