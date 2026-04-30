package com.webdinhduong.chatbot.repository;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.webdinhduong.chatbot.entity.User;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByUsername(String username);
}