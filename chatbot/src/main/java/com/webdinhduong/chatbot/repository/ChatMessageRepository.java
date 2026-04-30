package com.webdinhduong.chatbot.repository;

import com.webdinhduong.chatbot.entity.ChatMessage;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import java.util.List;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, Long> {

    List<ChatMessage> findTop50ByUserIdOrderByCreatedAtDesc(Long userId);

    List<ChatMessage> findBySessionIdOrderByCreatedAtAsc(String sessionId);

    @Query("SELECT m FROM ChatMessage m WHERE m.user.id = :userId AND m.role = 'user' " +
           "AND m.sessionId IS NOT NULL " +
           "AND m.createdAt = (SELECT MIN(m2.createdAt) FROM ChatMessage m2 " +
           "WHERE m2.sessionId = m.sessionId AND m2.role = 'user') " +
           "ORDER BY m.createdAt DESC")
    List<ChatMessage> findFirstUserMessagePerSession(@Param("userId") Long userId);
}
