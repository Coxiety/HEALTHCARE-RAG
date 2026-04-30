package com.webdinhduong.chatbot.config;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

@Configuration
public class SecurityConfig {

    // 1. Khai báo (Inject) bộ lọc JwtFilter vào đây
    @Autowired
    private JwtFilter jwtFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http.csrf(csrf -> csrf.disable())
            .authorizeHttpRequests(auth -> auth
                // Vẫn giữ nguyên cấu hình mở cửa cho thư mục tĩnh và API auth của bạn
                .requestMatchers("/", "/index.html", "/css/**", "/js/**", "/api/auth/**", "/favicon.ico").permitAll()
                .anyRequest().authenticated()
            )
            // 2. DÒNG QUAN TRỌNG NHẤT: Đặt JwtFilter đứng trước bộ lọc mặc định của Spring
            .addFilterBefore(jwtFilter, UsernamePasswordAuthenticationFilter.class);
            
        return http.build();
    }
}