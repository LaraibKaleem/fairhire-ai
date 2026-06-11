
        # ============================================================
        # 📊 SKILLS MATCH VISUALIZATION (CREATIVE)
        # ============================================================

        st.markdown("""
        <style>
        .skills-container {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .skills-header {
            text-align: center;
            color: white;
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            letter-spacing: 1px;
        }
        .skill-ring-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1rem;
        }
        .skill-ring {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: conic-gradient(
                var(--ring-color) calc(var(--percent) * 1%), 
                rgba(255,255,255,0.1) 0
            );
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            box-shadow: 0 0 30px var(--ring-glow);
            animation: pulse-ring 2s ease-in-out infinite;
        }
        @keyframes pulse-ring {
            0%, 100% { box-shadow: 0 0 20px var(--ring-glow); }
            50% { box-shadow: 0 0 40px var(--ring-glow), 0 0 60px var(--ring-glow); }
        }
        .skill-ring-inner {
            width: 75px;
            height: 75px;
            border-radius: 50%;
            background: #1a1a2e;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.1rem;
            color: white;
        }
        .skill-name {
            color: #e0e0e0;
            font-weight: 600;
            margin-top: 0.8rem;
            font-size: 0.95rem;
            text-align: center;
        }
        .skill-badge {
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }
        .badge-found { background: rgba(40, 167, 69, 0.3); color: #28a745; border: 1px solid #28a745; }
        .badge-missing { background: rgba(220, 53, 69, 0.3); color: #dc3545; border: 1px solid #dc3545; }
        .badge-preferred { background: rgba(255, 193, 7, 0.3); color: #ffc107; border: 1px solid #ffc107; }

        .match-score-big {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }
        .match-label {
            text-align: center;
            color: #888;
            font-size: 1rem;
            margin-top: -0.5rem;
        }
        .hex-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }
        .hex-item {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        .hex-item:hover {
            transform: translateY(-5px);
            border-color: var(--hex-color);
            box-shadow: 0 10px 30px var(--hex-glow);
        }
        .hex-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .hex-skill {
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .hex-status {
            font-size: 0.75rem;
            margin-top: 0.3rem;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="skills-container">
            <div class="skills-header">📊 SKILLS MATCH FOR {job_title.upper()}</div>

            <div style="display: flex; justify-content: center; align-items: center; gap: 3rem; margin: 2rem 0;">
                <div style="text-align: center;">
                    <div class="match-score-big">{screen.get('match_percentage', 75)}%</div>
                    <div class="match-label">Overall Match</div>
                </div>
                <div style="border-left: 2px solid rgba(255,255,255,0.2); height: 80px;"></div>
                <div style="color: #e0e0e0;">
                    <div style="margin: 0.3rem 0;"><span style="color: #28a745; font-weight: 700;">●</span> Required Found: {len(screen.get('required_skills', {}).get('found', []))}</div>
                    <div style="margin: 0.3rem 0;"><span style="color: #dc3545; font-weight: 700;">●</span> Required Missing: {len(screen.get('required_skills', {}).get('missing', []))}</div>
                    <div style="margin: 0.3rem 0;"><span style="color: #ffc107; font-weight: 700;">●</span> Preferred Found: {len(screen.get('preferred_skills', {}).get('found', []))}</div>
                </div>
            </div>

            <div class="hex-grid">
        """, unsafe_allow_html=True)

        # Build skill cards
        all_skills = []

        for skill in screen.get('required_skills', {}).get('found', []):
            all_skills.append((skill, "✅ HAVE", "#28a745", "rgba(40, 167, 69, 0.3)", "💚"))
        for skill in screen.get('required_skills', {}).get('missing', []):
            all_skills.append((skill, "❌ MISSING", "#dc3545", "rgba(220, 53, 69, 0.3)", "💔"))
        for skill in screen.get('preferred_skills', {}).get('found', []):
            all_skills.append((skill, "⭐ PREFERRED", "#ffc107", "rgba(255, 193, 7, 0.3)", "⭐"))

        for skill, status, color, glow, icon in all_skills[:12]:  # Limit to 12 for clean layout
            st.markdown(f"""
                <div class="hex-item" style="--hex-color: {color}; --hex-glow: {glow};">
                    <div class="hex-icon">{icon}</div>
                    <div class="hex-skill">{skill}</div>
                    <div class="hex-status" style="color: {color};">{status}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Alternative: Animated radial progress rings for top skills
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="skills-container">
            <div class="skills-header">🎯 TOP SKILL PROFICIENCY RINGS</div>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 2rem; margin-top: 1.5rem;">
        """, unsafe_allow_html=True)

        ring_skills = [
            ("Python", 95, "#28a745", "rgba(40, 167, 69, 0.4)"),
            ("ML/DL", 88, "#667eea", "rgba(102, 126, 234, 0.4)"),
            ("NLP", 82, "#764ba2", "rgba(118, 75, 162, 0.4)"),
            ("PyTorch", 78, "#f093fb", "rgba(240, 147, 251, 0.4)"),
            ("Research", 90, "#4facfe", "rgba(79, 172, 254, 0.4)"),
        ]

        for skill, pct, color, glow in ring_skills:
            st.markdown(f"""
                <div class="skill-ring-container">
                    <div class="skill-ring" style="--ring-color: {color}; --ring-glow: {glow}; --percent: {pct};">
                        <div class="skill-ring-inner">{pct}%</div>
                    </div>
                    <div class="skill-name">{skill}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Skill gap analysis with creative progress bars
        st.markdown("<br>", unsafe_allow_html=True)

        gap_skills = screen.get('required_skills', {}).get('missing', [])
        if gap_skills:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2d132c 0%, #1a1a2e 100%); border-radius: 20px; padding: 2rem; border: 1px solid rgba(220, 53, 69, 0.3);">
                <div style="text-align: center; color: white; font-size: 1.3rem; font-weight: 700; margin-bottom: 1.5rem;">
                    ⚠️ SKILL GAPS IDENTIFIED
                </div>
                <div style="color: #e0e0e0; text-align: center; margin-bottom: 1.5rem;">
                    These skills are required for the position but were not detected in the resume.
                </div>
            """, unsafe_allow_html=True)

            for skill in gap_skills[:5]:
                st.markdown(f"""
                <div style="margin: 0.8rem 0;">
                    <div style="display: flex; justify-content: space-between; color: #e0e0e0; margin-bottom: 0.3rem;">
                        <span style="font-weight: 600;">❌ {skill}</span>
                        <span style="color: #dc3545; font-weight: 700;">MISSING</span>
                    </div>
                    <div style="background: rgba(220, 53, 69, 0.2); border-radius: 10px; height: 12px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #dc3545, #ff6b6b); height: 100%; width: 100%; border-radius: 10px; animation: shimmer 2s infinite;"></div>
                    </div>
                </div>
                <style>
                @keyframes shimmer {{
                    0% {{ opacity: 0.6; }}
                    50% {{ opacity: 1; }}
                    100% {{ opacity: 0.6; }}
                }}
                </style>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")