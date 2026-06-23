package com.example.SprinAICode.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
public class OllamaAiController {

    @Autowired
    private VectorStore vectorStore;

    private final ChatClient chatClient;
    private final ChatMemory chatMemory = MessageWindowChatMemory.builder().build();

    @Autowired
    private EmbeddingModel embeddingModel;
    @Autowired
    private ChatModel chatModel;

    @PostMapping("/embeddings")
    public float[] testEmbedding(@RequestParam String text) {
        float[] vector = embeddingModel.embed(text);
        System.out.println("Embedding length: " + vector.length);
        return vector;
    }

    public OllamaAiController(OllamaChatModel chatModel) {

        this.chatClient = ChatClient.builder(chatModel)
                .defaultAdvisors(
                        MessageChatMemoryAdvisor.builder(chatMemory).build()
                )
                .build();
    }

    @GetMapping("/apiOllama/{message}")
    public ResponseEntity<String> getAnswer(@PathVariable String message) {

        ChatResponse response = chatClient.prompt()
                .user(message)
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, "default"))
                .call()
                .chatResponse();

        System.out.println("Content: " +
                response.getResult().getOutput().getText());

        System.out.println("Metadata: " +
                response.getMetadata()
                        .getModel());

        return ResponseEntity.ok(
                response.getResult().getOutput().getText()
        );
    }

    @PostMapping("/api/recommend")
    public ResponseEntity<String> recommend(@RequestParam String type,
                                            @RequestParam String year,
                                            @RequestParam String lang) {
        String tempt = """
                    I want to watchh a {type} movie with {year} {lang}.
                """;

        PromptTemplate promptTemplate = new PromptTemplate(tempt);

        Prompt prompt = promptTemplate.create(Map.of("type", type, "year", year, "lang", lang));

        String response = chatClient
                .prompt(prompt)
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, "recommend-session"))
                .call()
                .content();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/similarity")
    public ResponseEntity<Double> similarity(@RequestParam String type1, @RequestParam String type2) {
        float[] vector1 = embeddingModel.embed(type1);
        float[] vector2 = embeddingModel.embed(type2);

        double dotProduct = 0;
        double norm1 = 0;
        double norm2 = 0;

        for (int i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
            norm1 += Math.pow(vector1[i], 2);
            norm2 += Math.pow(vector2[i], 2);
        }

        return ResponseEntity.ok((dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2))));
    }

    @PostMapping("/product")
    public List<Document> getProducts(@RequestParam String txt) {

        float[] vector = embeddingModel.embed(txt);



        return vectorStore.similaritySearch(SearchRequest.builder().query(txt).topK(2).build());
    }

    @PostMapping("/apiOllama/ask")
    public ResponseEntity<String> getAnswerUsingrag(@RequestParam String query) {

        ChatResponse response =  ChatClient.builder(chatModel)
                .build().prompt()
                .advisors(QuestionAnswerAdvisor.builder(vectorStore).build())
                .user(query)
                .call()
                .chatResponse();

        return ResponseEntity.ok(response.getResult().getOutput().getText());
    }

}
