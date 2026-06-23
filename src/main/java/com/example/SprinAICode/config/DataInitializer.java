package com.example.SprinAICode.config;

import jakarta.annotation.PostConstruct;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

//import javax.swing.text.Document;
import java.util.List;

@Component
public class DataInitializer {

    @Autowired
    private VectorStore  vectorStore;
    @Autowired
    private JdbcTemplate jdbcTemplate;

    @PostConstruct
    public void initData(){

        //remove this if file size has changed
        //--------------------------------------------
        Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM vector_store",
                Integer.class
        );

        if (count != null && count > 0) {
            return; // Already loaded
        }

        //----------------------------------------------

        TextReader textReader = new TextReader(new ClassPathResource("productDetails.txt"));

        TokenTextSplitter splitter = TokenTextSplitter.builder().build();
        TokenTextSplitter splitter1 = TokenTextSplitter.builder()
                .withChunkSize(100)
                .withMinChunkSizeChars(30)
                .withMinChunkLengthToEmbed(5)
                .withMaxNumChunks(300)
                .build();
        List<Document> documents = splitter1.split(textReader.get());

        vectorStore.add(documents);
    }
}
