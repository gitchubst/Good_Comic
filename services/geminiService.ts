
import { GoogleGenAI, Part, HarmCategory, HarmBlockThreshold, Type } from "@google/genai";
import { ReferenceImage, GenerationOptions, StorySettings, StoryBlueprint, StoryScene } from '../types';

const getClient = () => {
  // 1. Check for manual override key first
  const customKey = localStorage.getItem('gemini_custom_api_key');
  if (customKey) {
      return new GoogleGenAI({ apiKey: customKey });
  }

  // 2. Fallback to environment/injected key (AI Studio / IDX)
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found. Please select a Google Cloud Project or enter a key manually.");
  }
  return new GoogleGenAI({ apiKey });
}

// Helper: Common safety settings to prevent blocking on creative content
const SAFETY_SETTINGS = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
];

export const generateImageWithGemini = async (
  prompt: string,
  referenceImages: ReferenceImage[],
  options: GenerationOptions
): Promise<string> => {
  const ai = getClient();
  const parts: Part[] = [];

  if (referenceImages && referenceImages.length > 0) {
    referenceImages.forEach(img => {
        parts.push({
            inlineData: {
                data: img.data,
                mimeType: img.mimeType,
            },
        });
    });
  }

  // Inject strict style and formatting constraints
  const layoutString = options.layout === '3x3' ? '3 columns and 3 rows (3x3 grid, 9 panels total)' : (options.layout === '2x2' ? '2 columns and 2 rows (2x2 grid, 4 panels total)' : (options.layout === '3x1' ? '3 columns and 1 row (3x1 grid, 3 panels total)' : '2 columns and 1 row (2x1 grid, 2 panels total)'));
  const technicalPrompt = `
  Requirement: High quality art. Adhere to panel layout. This image should be strict to the style of the reference image. The chapter number or page number should not be in the picture. The panels MUST be separated by a single thin black line. There MUST be absolutely NO white space, NO margins, and NO gutters between panels, and NO white space around the outer edges of the image. The panels must be completely flush and touch each other directly. The image should be 100% filled with the comic art, divided only by a 1-pixel black line. CRITICAL: DO NOT draw any speech bubbles, text, or words in the image. ABSOLUTE REQUIREMENT: The image MUST be a strict grid of exactly ${layoutString}. Do not deviate from this grid layout.
  `;

  const finalPrompt = `${prompt}\n${technicalPrompt}`;

  const aspectRatioConfig = (options.layout === '2x2' || options.layout === '3x3') ? "1:1" : "16:9";

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: { parts: [...parts, { text: finalPrompt }] },
      config: {
        imageConfig: {
          aspectRatio: aspectRatioConfig,
        },
        safetySettings: SAFETY_SETTINGS,
      }
    });

    const candidates = response.candidates;
    if (!candidates || candidates.length === 0) throw new Error("No candidates returned");

    const content = candidates[0].content;
    if (!content || !content.parts) throw new Error("Empty content");

    for (const part of content.parts) {
      if (part.inlineData && part.inlineData.data) {
        return part.inlineData.data;
      }
    }
    throw new Error("No image data found");

  } catch (error: any) {
    console.error("Gemini Generation Error:", error);
    if (error.message && (error.message.includes("Requested entity was not found") || error.message.includes("403") || error.message.includes("API Key"))) {
        throw new Error("API_KEY_INVALID");
    }
    throw new Error(error.message || "Failed to generate image.");
  }
};

// --- Story Mode Services ---

export const enhanceStoryConcept = async (concept: string): Promise<string> => {
  const ai = getClient();
  const systemPrompt = `You are a creative writing assistant.
  Task: Expand the story concept. 
  Style: Casual, natural, detailed.
  Input: "${concept}"
  Output: A single paragraph.`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: systemPrompt,
      config: {
        temperature: 0.9,
        topP: 0.95,
        safetySettings: SAFETY_SETTINGS,
      }
    });
    return response.text || "";
  } catch (error: any) {
    console.error("Enhancement Error:", error);
    throw new Error("Failed to enhance story.");
  }
};

const chunkArray = <T>(arr: T[], size: number): T[][] => {
  return Array.from({ length: Math.ceil(arr.length / size) }, (v, i) =>
    arr.slice(i * size, i * size + size)
  );
};

const cleanJson = (text: string) => {
    try {
        return JSON.parse(text);
    } catch (e) {
        const match = text.match(/```json\s*([\s\S]*?)\s*```/) || text.match(/```\s*([\s\S]*?)\s*```/);
        if (match) {
            return JSON.parse(match[1]);
        }
        throw new Error("Failed to parse JSON response: " + text.substring(0, 50) + "...");
    }
};

interface OutlineScene {
    title: string;
    synopsis: string;
    index: number;
    characterOverride?: string;
}

const generateOutline = async (settings: StorySettings): Promise<{ title: string, scenes: OutlineScene[] }> => {
    const ai = getClient();
    const charSummary = settings.characters.map(c => `${c.name} (${c.role})`).join(', ');

    let allScenes: OutlineScene[] = [];
    let storyTitle = settings.storyPrompt.substring(0, 50) || "Untitled Story";
    
    // Bypass AI outline generation for custom mode to preserve exact user text and scene count
    if (settings.sceneGenerationMode === 'custom' && settings.customSceneGuide) {
        for (let i = 0; i < settings.customSceneGuide.length; i++) {
            const guide = settings.customSceneGuide[i];
            allScenes.push({
                index: i + 1,
                title: `Scene ${i + 1}`,
                synopsis: (guide.type === 'custom' && guide.description.trim()) 
                    ? guide.description 
                    : `Continue the story naturally based on the concept: ${settings.storyPrompt}`,
                characterOverride: guide.characterOverride
            });
        }
        
        try {
            const titleResponse = await ai.models.generateContent({
                model: 'gemini-3-flash-preview',
                contents: `Generate a short, catchy title for a comic about: ${settings.storyPrompt}`,
                config: { safetySettings: SAFETY_SETTINGS }
            });
            if (titleResponse.text) storyTitle = titleResponse.text.replace(/["*]/g, '').trim();
        } catch (e) {}

        return { title: storyTitle, scenes: allScenes };
    }

    // Chunk outline generation to prevent output token limits on large scene counts
    const BATCH_SIZE = 10;
    const totalBatches = Math.ceil(settings.totalScenes / BATCH_SIZE);

    for (let i = 0; i < totalBatches; i++) {
        const startIdx = i * BATCH_SIZE; 
        const endIdx = Math.min((i + 1) * BATCH_SIZE, settings.totalScenes);
        const currentBatchCount = endIdx - startIdx;
        const startSceneNum = startIdx + 1;
        const endSceneNum = endIdx;

        const structurePrompt = `
        REQUIREMENT:
        Generate a structural outline for Scenes ${startSceneNum} to ${endSceneNum} (Batch ${i+1}/${totalBatches}).
        Total story length is ${settings.totalScenes} scenes.
        ${i === 0 ? "This is the beginning of the story." : "This is a continuation of the story. Maintain continuity."}
        ${i === totalBatches - 1 ? "This is the conclusion/end of the story." : ""}
        `;

        const isFirstBatch = i === 0;

        const prompt = `
        ACT AS A COMIC STORY ARCHITECT.
        STORY CONCEPT: "${settings.storyPrompt}"
        CHARACTERS: ${charSummary}
        
        ${structurePrompt}
        
        CRITICAL INSTRUCTION: You MUST generate EXACTLY ${currentBatchCount} scenes in the "scenes" array.
        Number them strictly from ${startSceneNum} to ${endSceneNum}.
        `;

        try {
            const response = await ai.models.generateContent({
                model: 'gemini-3-flash-preview',
                contents: prompt,
                config: { 
                    temperature: 0.9,
                    topP: 0.95,
                    responseMimeType: "application/json", 
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            title: { type: Type.STRING, description: "The title of the story" },
                            scenes: {
                                type: Type.ARRAY,
                                items: {
                                    type: Type.OBJECT,
                                    properties: {
                                        index: { type: Type.INTEGER },
                                        title: { type: Type.STRING },
                                        synopsis: { type: Type.STRING }
                                    },
                                    required: ["index", "title", "synopsis"]
                                }
                            }
                        },
                        required: ["scenes"]
                    },
                    safetySettings: SAFETY_SETTINGS 
                }
            });

            const data = cleanJson(response.text || "{}");
            
            if (isFirstBatch && data.title) {
                storyTitle = data.title;
            }

            if (data.scenes && Array.isArray(data.scenes)) {
                 // Validate and fix indices
                 const normalizedScenes = data.scenes.map((s: any, idx: number) => ({
                     ...s,
                     index: startSceneNum + idx
                 }));
                 allScenes = [...allScenes, ...normalizedScenes];
            }
        } catch (error) {
            console.error(`Error generating outline batch ${i}`, error);
            // Fallback for failed batch will be handled by filling missing scenes below
        }
    }

    // Fallback: Fill missing scenes if generation failed
    if (allScenes.length < settings.totalScenes) {
        const missing = settings.totalScenes - allScenes.length;
        const startFill = allScenes.length + 1;
        for (let k = 0; k < missing; k++) {
             allScenes.push({
                index: startFill + k,
                title: `Scene ${startFill + k}`,
                synopsis: "Scene content auto-filled due to generation interruption."
            });
        }
    }

    return { title: storyTitle, scenes: allScenes };
};

const FEMININE_HAIRSTYLES = [
    "long flowing hair", "messy bun", "ponytail", "braided hair", "twin tails", "bob cut", "wavy shoulder-length hair", "curly long hair", "straight long hair with bangs", "elegant updo", "half-up half-down hair", "loose curls", "side braid"
];

const CLOTHING_ITEMS = [
    "casual t-shirt and jeans", "elegant summer dress", "business suit", "cozy oversized sweater and leggings", "leather jacket and ripped jeans", "athletic wear", "gothic lolita dress", "bohemian skirt and crop top", "school uniform", "winter coat and scarf", "pajamas", "evening gown", "denim overalls", "swimsuit with a sarong", "sundress", "turtleneck and pleated skirt"
];

const getRandomItem = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];

const generateSceneBatch = async (
    outlineScenes: OutlineScene[], 
    settings: StorySettings,
    storyTitle: string
): Promise<StoryScene[]> => {
    const ai = getClient();
    const isBW = settings.colorMode === 'bw';

    const characterRules = settings.characters.map(c => {
        const hairPart = c.hairColor ? `${c.hairColor} hair, ` : '';
        const visualPart = c.hardDescription ? `${c.hardDescription} ` : '';
        const fullName = `${hairPart}${visualPart}${c.name}`.replace(/\s+/g, ' ').trim();
        return { originalName: c.name, enforcedName: fullName, bio: c.description };
    });

    const namingInstructions = characterRules.map(r => 
        `- ${r.originalName}: MUST refer to as "${r.enforcedName}"`
    ).join('\n');

    const layoutMode = settings.layoutMode || '2x2';
    let panelCount = 4;
    let layoutDescription = "";
    if (layoutMode === '3x3') {
        panelCount = 9;
        layoutDescription = "3x3 grid. It should strictly have 9 panels exactly (nothing more, nothing less) and it should be exactly 3 blocks long and exactly 3 blocks tall. The blocks should be next to each other in 3 rows of 3 blocks each.";
    } else if (layoutMode === '2x2') {
        panelCount = 4;
        layoutDescription = "2x2 grid. It should strictly have 4 panels exactly (nothing more, nothing less) and it should be exactly 2 blocks long and exactly 2 blocks tall. The blocks should be next to each other and be 2 blocks on one row and then a second row right bellow it of 2 blocks again to make it 2 blocks long and 2 blocks tall.";
    } else if (layoutMode === '2x1') {
        panelCount = 2;
        layoutDescription = "2x1 grid. It should strictly have 2 panels exactly (nothing more, nothing less) and it should be exactly 2 blocks long and exactly 1 block tall. The blocks should be next to each other on one row.";
    } else if (layoutMode === '3x1') {
        panelCount = 3;
        layoutDescription = "3x1 grid. It should strictly have 3 panels exactly (nothing more, nothing less) and it should be exactly 3 blocks long and exactly 1 block tall. The blocks should be next to each other on one row.";
    }

    let dialogueRules = "";
    if (settings.dialogueLevel === 'High') {
        dialogueRules = `EXTREME DIALOGUE DENSITY (High). ${panelCount} out of ${panelCount} panels MUST have dialogue. 4-5 short sentences per bubble. The dialogue MUST sound completely natural, conversational, and human. Avoid robotic, monotonic, or overly formal phrasing. Use contractions, slang where appropriate, and natural speech patterns. Make it sound like real people talking.`;
    } else if (settings.dialogueLevel === 'Medium') {
        dialogueRules = `VERY HIGH DIALOGUE DENSITY (Medium). ${panelCount} out of ${panelCount} panels MUST have dialogue. 1-3 short sentences. The dialogue MUST sound completely natural, conversational, and human. Avoid robotic, monotonic, or overly formal phrasing. Use contractions, slang where appropriate, and natural speech patterns. Make it sound like real people talking.`;
    } else {
        dialogueRules = `MODERATE DIALOGUE DENSITY. ${Math.max(1, Math.floor(panelCount / 2))} panels with dialogue. The dialogue MUST sound completely natural, conversational, and human. Avoid robotic, monotonic, or overly formal phrasing. Use contractions, slang where appropriate, and natural speech patterns. Make it sound like real people talking.`;
    }
    dialogueRules += " CRITICAL: Format dialogue strictly at the end of each panel's description like this: | Dialogue: \"The text here\"";

    let visualSpecificityRules = "";
    if (isBW) {
        visualSpecificityRules = `
         a) CLOTHING: In EVERY panel description, specify exactly what each clothing piece looks like. DO NOT include any colors in the description.
         b) SETTING: In EVERY panel description, provide more details on the location/background. DO NOT include any colors.
        `;
    } else {
        visualSpecificityRules = `
         a) CLOTHING: In EVERY panel description, specify exactly what each clothing piece looks like. Be specific in every scene.
         b) SETTING: In EVERY panel description, provide more details on the location/background.
        `;
    }

    const sharedInstructions = "this image should be strict to the style of the reference image. The chapter number or page number should not be in the picture. The panels MUST be separated by a single thin black line. There MUST be absolutely NO white space, NO margins, and NO gutters between panels, and NO white space around the outer edges of the image. The panels must be completely flush and touch each other directly. The image should be 100% filled with the comic art, divided only by a 1-pixel black line. CRITICAL: DO NOT draw any speech bubbles, text, or words in the image.";

    const templatePrefix = isBW 
        ? `Make a comic in the same art style as the given reference images (the character designs should be strictly based on the reference images, especially the face, while the clothes should not need to be based on the reference images while the body type should be the same as the reference images except for the possibility of a pregnant belly) and make your image only black and white (this comic should be only black and white), with exactly ${panelCount} panels in a ${layoutDescription} Make sure the only colors in this comic are black and white. There should be no color in this. The topic should be different though than the png/image, but just the art style should be the same. ${sharedInstructions}`
        : `Make a comic in the same art style as the given reference images (the character designs should be strictly based on the reference images, especially the face, while the clothes should not need to be based on the reference images while the body type should be the same as the reference images except for the possibility of a pregnant belly), but colored, with exactly ${panelCount} panels in a ${layoutDescription} The topic should be different though than the png/image, but just the art style should be the same. ${sharedInstructions}`;

    const sceneDescriptions = outlineScenes.map(os => {
        let overridePrompt = "";
        
        settings.characters.forEach(c => {
            let hasOverride = false;
            let hairPart = c.hairColor ? `${c.hairColor} hair, ` : '';
            let hairStylePart = c.randomizeHairStyle ? `${getRandomItem(FEMININE_HAIRSTYLES)}, ` : '';
            let clothingPart = c.randomizeClothing ? `wearing ${getRandomItem(CLOTHING_ITEMS)}, ` : '';
            
            let hardDescPart = c.hardDescription ? `${c.hardDescription} ` : '';
            if (c.role === 'Main' && os.characterOverride && os.characterOverride.trim() !== "") {
                hardDescPart = `${os.characterOverride} `;
                hasOverride = true;
            }
            
            if (c.randomizeHairStyle || c.randomizeClothing || hasOverride) {
                const forcedName = `${hairPart}${hairStylePart}${clothingPart}${hardDescPart}${c.name}`.replace(/\s+/g, ' ').trim();
                overridePrompt += `\n[STRICT VISUAL RULE FOR THIS SCENE ONLY]: For ${c.name}, you MUST use the physical description: "${forcedName}". This replaces their default description for this scene.`;
            }
        });

        return `Scene ${os.index}: ${os.title}. Synopsis: ${os.synopsis}.${overridePrompt}`;
    }).join('\n\n');

    const batchPrompt = `
    ACT AS A STRICT COMIC SCRIPTWRITING ENGINE.
    STORY TITLE: ${storyTitle}
    
    TASK: Generate full scripts for the following ${outlineScenes.length} scenes:
    ${sceneDescriptions}
    
    RULES:
    1. EACH scene must have EXACTLY ${settings.imagesPerScene} "pages" (prompt strings).
    2. STRICTLY FOLLOW NAMING (unless overriden in scene-specific rules above):
    ${namingInstructions}
    3. DIALOGUE: ${dialogueRules}
    4. PROMPT STRUCTURE:
       Each 'page' string MUST start with: "${templatePrefix}"
       Then immediately follow with: "Scene {N}: {Title}. Setting: {Detailed Setting}. Characters: {List}."
       Then: "Panel Breakdown: P1: {Content} P2: {Content} ... P${panelCount}: {Content}"
    5. VISUAL SPECIFICITY:
    ${visualSpecificityRules}
    
    CRITICAL INSTRUCTION: You MUST return EXACTLY ${outlineScenes.length} scene(s) in the JSON array.
    `;

    try {
        const response = await ai.models.generateContent({
            model: 'gemini-3-flash-preview',
            contents: batchPrompt,
            config: { 
                temperature: 0.9,
                topP: 0.95,
                responseMimeType: "application/json", 
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        scenes: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.OBJECT,
                                properties: {
                                    index: { type: Type.INTEGER },
                                    title: { type: Type.STRING },
                                    pages: {
                                        type: Type.ARRAY,
                                        items: { type: Type.STRING }
                                    }
                                },
                                required: ["index", "title", "pages"]
                            }
                        }
                    },
                    required: ["scenes"]
                },
                safetySettings: SAFETY_SETTINGS 
            }
        });
        
        if (!response.text) {
            throw new Error("Empty content received from Gemini API");
        }
        
        const data = cleanJson(response.text);
        
        if (!data.scenes || !Array.isArray(data.scenes) || data.scenes.length === 0) {
            throw new Error("No scenes generated in the response");
        }
        
        return data.scenes.map((s: any) => ({
            id: crypto.randomUUID(),
            title: s.title,
            pages: (s.pages || []).map((p: string, pIdx: number) => ({
                id: crypto.randomUUID(),
                globalIndex: 0,
                sceneIndex: s.index,
                pageInSceneIndex: pIdx + 1,
                prompt: p
            }))
        }));

    } catch (e: any) {
        console.error("Batch generation failed:", e);
        throw e;
    }
};

export const generateStoryScript = async (settings: StorySettings): Promise<StoryBlueprint> => {
  try {
    const outline = await generateOutline(settings);
    // Process 1 scene at a time to prevent token limits and ensure exact counts
    const batchSize = 1; 
    const sceneChunks = chunkArray(outline.scenes, batchSize);
    
    let allScenes: StoryScene[] = [];
    for (const chunk of sceneChunks) {
        let retries = 2;
        let success = false;
        while (retries > 0 && !success) {
            try {
                const chunkScenes = await generateSceneBatch(chunk, settings, outline.title);
                allScenes = [...allScenes, ...chunkScenes];
                success = true;
            } catch (e) {
                retries--;
                if (retries === 0) {
                    console.error("Failed chunk after retries", chunk);
                    throw e;
                }
                await new Promise(r => setTimeout(r, 1000));
            }
        }
    }
    
    let globalCounter = 1;
    allScenes.sort((a, b) => {
        const indexA = parseInt(a.pages[0]?.prompt?.match(/Scene (\d+):/)?.[1] || "0") || 0;
        const indexB = parseInt(b.pages[0]?.prompt?.match(/Scene (\d+):/)?.[1] || "0") || 0;
        return indexA - indexB;
    });

    if (settings.sceneGenerationMode === 'custom' && settings.customSceneGuide) {
        allScenes.forEach((scene, idx) => {
            if (settings.customSceneGuide && settings.customSceneGuide[idx]) {
                scene.referenceImage = settings.customSceneGuide[idx].referenceImage;
            }
        });
    }

    allScenes.forEach((scene, sIdx) => {
        scene.pages.forEach((page, pIdx) => {
            page.globalIndex = globalCounter++;
            page.sceneIndex = sIdx + 1;
            page.pageInSceneIndex = pIdx + 1;
        });
    });

    return {
        title: outline.title,
        scenes: allScenes
    };
  } catch (error: any) {
    console.error("Story Script Generation Error:", error);
    throw new Error("Failed to generate story script: " + error.message);
  }
};

export const generateAdditionalScenes = async (
    settings: StorySettings,
    newSceneDescription: string,
    count: number,
    currentBlueprint: StoryBlueprint,
    insertionIndex: number,
    characterOverride?: string
): Promise<StoryScene[]> => {
    const ai = getClient();
    const charSummary = settings.characters.map(c => `${c.name} (${c.role})`).join(', ');

    const outlinePrompt = `
    ACT AS A COMIC STORY ARCHITECT. ADDING SCENES TO EXISTING STORY.
    EXISTING STORY TITLE: "${currentBlueprint.title}"
    CHARACTERS: ${charSummary}
    TASK: Create ${count} NEW scenes.
    CONTENT OF NEW SCENES: "${newSceneDescription}"
    `;

    try {
        const response = await ai.models.generateContent({
            model: 'gemini-3-flash-preview',
            contents: outlinePrompt,
            config: { 
                temperature: 0.9,
                topP: 0.95,
                responseMimeType: "application/json", 
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        scenes: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.OBJECT,
                                properties: {
                                    index: { type: Type.INTEGER },
                                    title: { type: Type.STRING },
                                    synopsis: { type: Type.STRING }
                                },
                                required: ["index", "title", "synopsis"]
                            }
                        }
                    },
                    required: ["scenes"]
                },
                safetySettings: SAFETY_SETTINGS 
            }
        });
        const data = cleanJson(response.text || "{}");
        
        let outlineScenes = data.scenes || [];
        if (characterOverride && characterOverride.trim() !== "") {
            outlineScenes = outlineScenes.map((s: any) => ({
                ...s,
                characterOverride: characterOverride
            }));
        }
        
        return await generateSceneBatch(outlineScenes, settings, currentBlueprint.title);
    } catch (error: any) {
        throw new Error("Failed to generate additional scenes: " + error.message);
    }
};

export const refineStoryPrompts = async (
  currentBlueprint: StoryBlueprint,
  instruction: string,
  scope: 'story' | 'scene' | 'page',
  targetId?: string
): Promise<StoryBlueprint> => {
  const ai = getClient();
  const systemPrompt = `
  You are a JSON editor helper.
  Task: Modify the Comic Script JSON based on: "${instruction}".
  Scope: ${scope} ${targetId ? `(Target ID: ${targetId})` : ''}
  RULES:
  1. Only edit 'prompt' strings.
  2. Maintain JSON structure and existing format.
  3. Ensure constraints about reference style and panel lines remain in the prompts.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: systemPrompt + "\n\nINPUT JSON:\n" + JSON.stringify(currentBlueprint),
      config: {
        temperature: 0.9,
        topP: 0.95,
        responseMimeType: "application/json",
        responseSchema: {
            type: Type.OBJECT,
            properties: {
                title: { type: Type.STRING },
                scenes: {
                    type: Type.ARRAY,
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            id: { type: Type.STRING },
                            title: { type: Type.STRING },
                            referenceImage: {
                                type: Type.OBJECT,
                                properties: {
                                    data: { type: Type.STRING },
                                    mimeType: { type: Type.STRING }
                                }
                            },
                            pages: {
                                type: Type.ARRAY,
                                items: {
                                    type: Type.OBJECT,
                                    properties: {
                                        id: { type: Type.STRING },
                                        globalIndex: { type: Type.INTEGER },
                                        sceneIndex: { type: Type.INTEGER },
                                        pageInSceneIndex: { type: Type.INTEGER },
                                        prompt: { type: Type.STRING }
                                    },
                                    required: ["id", "globalIndex", "sceneIndex", "pageInSceneIndex", "prompt"]
                                }
                            }
                        },
                        required: ["id", "title", "pages"]
                    }
                }
            },
            required: ["title", "scenes"]
        },
        safetySettings: SAFETY_SETTINGS,
      }
    });
    return JSON.parse(response.text || "{}") as StoryBlueprint;
  } catch (error: any) {
    throw new Error("Failed to refine script: " + error.message);
  }
};
