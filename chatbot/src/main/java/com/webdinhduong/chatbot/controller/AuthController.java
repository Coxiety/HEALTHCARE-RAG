package com.webdinhduong.chatbot.controller;

import com.webdinhduong.chatbot.entity.User;
import com.webdinhduong.chatbot.repository.UserRepository;
import com.webdinhduong.chatbot.util.JwtUtil; // Bạn cần tạo file này như tôi đã hướng dẫn trước đó
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/auth")
@CrossOrigin(origins = "*")
public class AuthController {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private JwtUtil jwtUtil;

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody User user) {
        if(userRepository.findByUsername(user.getUsername()).isPresent()) {
            return ResponseEntity.badRequest().body("Tên đăng nhập đã tồn tại!");
        }
        userRepository.save(user);
        return ResponseEntity.ok("Đăng ký thành công!");
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody User user) {
        Optional<User> dbUser = userRepository.findByUsername(user.getUsername());
        
        // Kiểm tra mật khẩu (Lưu ý: Thực tế nên dùng BCrypt)
        if(dbUser.isPresent() && dbUser.get().getPassword().equals(user.getPassword())) {
            // Tạo Token bảo mật
            String token = jwtUtil.generateToken(dbUser.get().getUsername());
            
            // Trả về một Object chứa cả Token và thông tin User
            return ResponseEntity.ok(Map.of(
                "token", token,
                "userId", dbUser.get().getId(),
                "username", dbUser.get().getUsername(),
                "fullName", dbUser.get().getFullName()
            ));
        }
        return ResponseEntity.status(401).body("Sai tài khoản hoặc mật khẩu!");
    }
}