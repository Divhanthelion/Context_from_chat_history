use std::fs::{self, File, create_dir_all};
use std::path::{Path, PathBuf};
use std::collections::VecDeque;
use std::error::Error;
use std::io::{Read, Write};
use reqwest;
use chrono::Utc;
use serde::{Serialize, Deserialize};
use serde_json;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsModel, 
    SentenceEmbeddingsModelType,
    SentenceEmbeddingsConfig
};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Memory {
    id: usize,
    role: String,
    content: String,
    timestamp: String,
    metadata: serde_json::Value,
    embedding: Vec<f32>,
}

struct PersistentMemory {
    memory_path: PathBuf,
    vector_dimensions: usize,
    max_memories: usize,
    memories: VecDeque<Memory>,
    embedding_model: SentenceEmbeddingsModel,
}

impl PersistentMemory {
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        
        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            0.0
        } else {
            dot_product / (magnitude_a * magnitude_b)
        }
    }

    fn new(
        memory_path: &str, 
        vector_dimensions: usize,
        max_memories: usize,
    ) -> Result<Self, Box<dyn Error>> {
        // Create memory directory
        create_dir_all(memory_path)?;

        // Initialize memory database paths
        let memory_data_path = Path::new(memory_path).join("memory_data.json");

        // Load or create memory storage
        let memories = if memory_data_path.exists() {
            Self::load_memory_storage(&memory_data_path)?
        } else {
            VecDeque::new()
        };

        // Initialize embedding model
        let config = SentenceEmbeddingsConfig::from(SentenceEmbeddingsModelType::AllMiniLmL6V2);
        let embedding_model = SentenceEmbeddingsModel::new(config)?;

        Ok(Self {
            memory_path: memory_path.into(),
            vector_dimensions,
            max_memories,
            memories,
            embedding_model,
        })
    }

    fn load_memory_storage(
        data_path: &Path
    ) -> Result<VecDeque<Memory>, Box<dyn Error>> {
        // Load memories from JSON
        let mut file = File::open(data_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let memories: VecDeque<Memory> = serde_json::from_str(&contents)?;

        Ok(memories)
    }

    fn save_memory_storage(&self) -> Result<(), Box<dyn Error>> {
        // Save memories to JSON
        let memories_json = serde_json::to_string_pretty(&self.memories)?;
        let memory_data_path = self.memory_path.join("memory_data.json");
        fs::write(memory_data_path, memories_json)?;

        Ok(())
    }

    fn get_embedding(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        // Generate embedding using sentence transformer
        let embeddings = self.embedding_model.encode(&[text])?;
        Ok(embeddings[0].to_vec())
    }

    fn store_memory(
        &mut self, 
        role: &str, 
        content: &str, 
        metadata: Option<serde_json::Value>
    ) -> Result<(), Box<dyn Error>> {
        if content.trim().is_empty() {
            return Ok(());
        }

        let embedding = self.get_embedding(content)?;
        let timestamp = Utc::now().to_rfc3339();

        let memory = Memory {
            id: self.memories.len(),
            role: role.to_string(),
            content: content.to_string(),
            timestamp,
            metadata: metadata.unwrap_or(serde_json::json!({})),
            embedding,
        };

        // Add memory to memories
        self.memories.push_back(memory);

        // Maintain maximum memories limit
        if self.memories.len() > self.max_memories {
            let num_to_remove = self.memories.len() - self.max_memories;
            for _ in 0..num_to_remove {
                self.memories.pop_front();
            }
        }

        self.save_memory_storage()?;

        Ok(())
    }

    fn retrieve_relevant_memories(
        &self, 
        query: &str, 
        k: usize
    ) -> Result<Vec<Memory>, Box<dyn Error>> {
        if self.memories.is_empty() {
            return Ok(vec![]);
        }

        let query_embedding = self.get_embedding(query)?;
        let k = k.min(self.memories.len());

        // Compute cosine similarities and sort
        let mut scored_memories: Vec<(f32, Memory)> = self.memories
            .iter()
            .map(|memory| {
                let similarity = Self::cosine_similarity(&query_embedding, &memory.embedding);
                (similarity, memory.clone())
            })
            .collect();

        // Sort by similarity in descending order
        scored_memories.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Return top k memories
        Ok(scored_memories
            .into_iter()
            .take(k)
            .map(|(_, memory)| memory)
            .collect())
    }

    fn format_context(
        &self, 
        relevant_memories: &[Memory], 
        max_tokens: usize
    ) -> String {
        let context_parts: Vec<String> = relevant_memories
            .iter()
            .map(|memory| format!("{}: {}", memory.role.to_uppercase(), memory.content))
            .collect();

        let context = context_parts.join("\n");

        // Simple truncation
        if context.len() > max_tokens * 4 {
            format!("{}...", &context[..max_tokens * 4])
        } else {
            context
        }
    }
}
    #[derive(Serialize)]
    struct OllamaRequest {
        model: String,
        prompt: String,
        stream: bool,
    }
    
    #[derive(Deserialize)]
    struct OllamaResponse {
        response: String,
    }
    
    struct LocalLLMWithMemory {
        memory: PersistentMemory,
        model_name: String,
    }
    
    impl LocalLLMWithMemory {
        fn new(llm_model_name: &str, memory_path: &str) -> Result<Self, Box<dyn Error>> {
            let memory = PersistentMemory::new(memory_path, 384, 1000)?;
    
            // Verify the model is available in Ollama
            Self::check_model_availability(llm_model_name)?;
    
            Ok(Self {
                memory,
                model_name: llm_model_name.to_string(),
            })
        }
    
        fn check_model_availability(model_name: &str) -> Result<(), Box<dyn Error>> {
            let client = reqwest::blocking::Client::new();
            let response = client.get("http://localhost:11434/api/tags")
                .send()?
                .json::<serde_json::Value>()?;
    
            let models = response["models"].as_array().ok_or("Failed to parse models")?;
            let model_exists = models.iter().any(|model| {
                model["name"].as_str().map_or(false, |name| 
                    name == model_name || name.starts_with(&format!("{}:", model_name))
                )
            });
    
            if !model_exists {
                return Err(format!("Model {} not found in Ollama", model_name).into());
            }
    
            Ok(())
        }
    
        fn chat(&mut self, user_input: &str) -> Result<String, Box<dyn Error>> {
            // Store user input in memory
            self.memory.store_memory("user", user_input, None)?;
    
            // Retrieve relevant memories
            let relevant_memories = self.memory.retrieve_relevant_memories(user_input, 5)?;
    
            // Format memories as context
            let memory_context = self.memory.format_context(&relevant_memories, 1024);
    
            // Construct prompt with memory context
            let prompt = format!(
                r#"The following is relevant context from previous conversations:
    
    {}
    
    Current conversation:
    USER: {}
    ASSISTANT:"#,
                memory_context, user_input
            );
    
            // Prepare Ollama API request
            let client = reqwest::blocking::Client::new();
            let request_body = OllamaRequest {
                model: self.model_name.clone(),
                prompt,
                stream: false,
            };
    
            // Send request to Ollama
            let response = client.post("http://localhost:11434/api/generate")
                .json(&request_body)
                .send()?
                .json::<OllamaResponse>()?;
    
            let response_text = response.response.trim().to_string();
    
            // Store response in memory
            self.memory.store_memory("assistant", &response_text, None)?;
    
            Ok(response_text)
        }
    }
    
    fn main() -> Result<(), Box<dyn Error>> {
        // Allow model selection from command line or default to llama3.2
        let model_name = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "llama3.2".to_string());
    
        // Initialize local LLM with memory
        let mut llm = LocalLLMWithMemory::new(
            &model_name, 
            "memory"
        )?;
    
        // Interactive chat loop
        println!("Chat with your memory-enhanced LLM using {} (type 'exit' to quit)", model_name);
        loop {
            let mut user_input = String::new();
            print!("You: ");
            std::io::stdout().flush()?;
            std::io::stdin().read_line(&mut user_input)?;
    
            let user_input = user_input.trim();
            if user_input == "exit" {
                break;
            }
    
            match llm.chat(user_input) {
                Ok(response) => println!("LLM: {}", response),
                Err(e) => eprintln!("Error: {}", e),
            }
        }
    
        Ok(())
    }