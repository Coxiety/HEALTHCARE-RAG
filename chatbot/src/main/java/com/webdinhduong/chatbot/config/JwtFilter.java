package com.webdinhduong.chatbot.config;

import java.io.IOException;
import java.util.ArrayList;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;

import com.webdinhduong.chatbot.util.JwtUtil;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@Component
public class JwtFilter extends OncePerRequestFilter {

    @Autowired
    private JwtUtil jwtUtil;

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        
        try {
            // 1. Lấy chuỗi Authorization từ Header của Request
            String headerAuth = request.getHeader("Authorization");
            String token = null;

            // 2. Kiểm tra xem Header có bắt đầu bằng "Bearer " không (chuẩn JWT)
            if (StringUtils.hasText(headerAuth) && headerAuth.startsWith("Bearer ")) {
                token = headerAuth.substring(7); // Cắt bỏ 7 ký tự đầu "Bearer " để lấy Token
            }

            // 3. Nếu có Token và Token đó hợp lệ (dùng hàm validateToken đã thêm ở JwtUtil)
            if (token != null && jwtUtil.validateToken(token)) {
                String username = jwtUtil.extractUsername(token);
                
                // 4. Tạo đối tượng xác thực và lưu vào SecurityContext của Spring
                UsernamePasswordAuthenticationToken authentication = 
                        new UsernamePasswordAuthenticationToken(username, null, new ArrayList<>());
                
                authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                
                // Đánh dấu người dùng này đã đăng nhập thành công cho phiên làm việc này
                SecurityContextHolder.getContext().setAuthentication(authentication);
            }
        } catch (Exception e) {
            // Log lỗi nếu có vấn đề trong quá trình giải mã
            logger.error("Không thể xác thực người dùng: {}", e);
        }

        // 5. Cho phép Request tiếp tục đi tới Controller
        filterChain.doFilter(request, response);
    }
}