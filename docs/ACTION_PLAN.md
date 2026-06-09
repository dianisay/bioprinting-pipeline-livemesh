# Plan de Acción — Tesis Doctoral Diana Ayala Roldán

## Estado Actual (Diagnóstico)

### Lo que YA está hecho
- Escritura completa de los 6 capítulos (estructura, narrativa, ecuaciones, tablas comparativas)
- Abstract en inglés y español (con placeholders numéricos)
- References.bib con ~90 entradas (algunas con autores genéricos que corregir)
- Diseño metodológico completo de los 6 módulos

### Lo que FALTA (todo lo crítico)
- **104 valores numéricos en rojo** en Results que necesitan datos reales
- **34 valores en rojo** en Discussion que dependen de los de Results
- **11 valores en rojo** en Conclusions
- **23 figuras PLACEHOLDER** que necesitan imágenes reales de experimentos
- **Implementación completa** de todo el código (CNN, Transformer, decoders, pipeline)
- **Experimentos** que generen los datos reales
- **5 notas "NOTE TO FUTURE ME"** pendientes

---

## Fases de Trabajo

---

### FASE 0: Setup del Entorno (1 semana)

**Objetivo:** Tener todo listo para programar sin fricción.

- [ ] **0.1** Crear repositorio Git para el código de la tesis
- [ ] **0.2** Configurar entorno Python (PyTorch, torchvision, matplotlib, scipy, roboticstoolbox-python, PuLP, Open3D)
- [ ] **0.3** Configurar CoppeliaSim + ZeroMQ remote API (Python)
- [ ] **0.4** Descargar y preparar dataset FUSeg (1,210 imágenes, usar las 934 filtradas)
- [ ] **0.5** Verificar que el modelo URDF del UR5 + gantry funciona en CoppeliaSim
- [ ] **0.6** Definir estructura de carpetas del proyecto:
  ```
  /code
    /module1_boundary_detection
    /module2_3d_reconstruction
    /module3_trajectory_generation
    /module4_robot_planning
    /module5_execution
    /module6_validation
    /poc_baseline
    /data
    /results
    /figures
  ```

---

### FASE 1: PoC Baseline — U-Net + G-Code (COMPLETADA)

**Estado:** COMPLETADA durante 2025.

Incluye: U-Net entrenada en FUSeg, pipeline G-Code funcional, simulación en CoppeliaSim con IK y PID. Los resultados del PoC sirven como baseline de comparación para el sistema propuesto (Tabla 4.11 del documento).

**Pendiente de esta fase:** Verificar que los datos del PoC ya estén reflejados en los placeholders de la Sección 4.1, o extraer las métricas finales si aún no se han sustituido.

---

### FASE 2: Módulo 1 — CNN-Transformer con Polar Decoder (4-5 semanas)

**Objetivo:** El corazón computacional de la tesis. Implementar la arquitectura desde cero y correr el ablation study.

**Prioridad:** CRÍTICA — es LA contribución principal.

#### 2.1 Preparación de Datos
- [ ] Convertir máscaras FUSeg a ground-truth polar: centroide + N radios equiangulares
- [ ] Implementar generador de wounds sintéticas (2,000 imágenes): formas star-convex aleatorias sobre texturas de piel
- [ ] Verificar que la conversión polar↔cartesiano es exacta (test unitario)
- [ ] Split train/val/test (ej. 70/15/15 o similar)

#### 2.2 Encoder (ResNet-50 + Transformer)
- [ ] Implementar ResNet-50 backbone (pretrained ImageNet) → feature tensor 16x16x2048
- [ ] 1x1 Conv projection → 16x16x256
- [ ] Learned 2D positional encoding (256-dim)
- [ ] Transformer encoder: 6 layers, 8 heads, d=256
- [ ] Verificar shapes en cada paso con un forward pass de prueba

#### 2.3 Decoder v3 — Polar (propuesto)
- [ ] Centroid head: global average pooling → FC → (x_c, y_c)
- [ ] Radii head: global average pooling → FC → N radios
- [ ] Conversión polar→cartesiano (Eq. 2 del capítulo)
- [ ] Implementar loss combinada: L_centroid + L_radii + L_points (con λ_c=1, λ_r=1, λ_p=0.5)

#### 2.4 Decoder v1 — Parallel Cartesian (DETR-style)
- [ ] N learned query vectors + cross-attention decoder
- [ ] Hungarian matching loss
- [ ] Mismo encoder que v3

#### 2.5 Decoder v2 — Autoregressive Cartesian
- [ ] Transformer decoder autoregresivo (teacher forcing en entrenamiento)
- [ ] Mismo encoder que v3

#### 2.6 Entrenamiento y Ablation
- [ ] Entrenar v1, v2, v3 con hiperparámetros idénticos (Adam, lr=1e-4, batch 8, max 100 epochs, early stopping patience=10)
- [ ] **DATOS A EXTRAER (reemplazan los rojos del ablation en Ch.4):**
  - Para cada decoder: Chamfer dist, Hausdorff dist, IoU, closure error, ordering %
  - Training/val loss curves (total + 3 componentes para v3)
  - Epoch de convergencia
  - Grid cualitativo de predicciones (buenas + failure cases)
  - Análisis del ~8% star-convex violations

---

### FASE 3: Módulo 2 — Reconstrucción 3D (2 semanas)

**Objetivo:** Reconstruir superficie 3D de herida desde múltiples vistas en simulación.

**Prioridad:** ALTA — alimenta todo Module 3.

- [ ] Configurar cámara virtual eye-in-hand en CoppeliaSim
- [ ] Crear 20 modelos de wound sintéticos (meshes 3D con geometría variable)
- [ ] Implementar adquisición multi-view: robot se posiciona en 5 viewpoints, captura RGB
- [ ] Obtener poses de cámara vía FK del robot
- [ ] Implementar pipeline MVS (o usar librería existente como OpenCV stereo / Open3D)
- [ ] Aplicar boundary masking de Module 1 para filtrar background
- [ ] **DATOS A EXTRAER:**
  - Mean/Max surface RMS error
  - Mean depth MAE
  - Reconstruction completeness %
  - Error con y sin boundary masking
  - Visualizaciones: GT mesh vs reconstructed vs error heatmap

---

### FASE 4: Módulo 3 — Generación de Trayectoria 3D (3 semanas)

**Objetivo:** Conformal honeycomb + TSP + toolpath completo.

**Prioridad:** ALTA — segunda contribución principal.

#### 4.1 Conformal Mapping
- [ ] Implementar Kasa circle fit para superficies cilíndricas (numpy/scipy)
- [ ] Implementar conformal map: superficie 3D → rectángulo (u,v)
- [ ] Implementar tangent-plane fallback para superficies planas
- [ ] Verificar distorsión angular (target: <3° para cilindros, <8° en zonas de alta curvatura)

#### 4.2 Honeycomb Lattice
- [ ] Generar grid hexagonal en dominio (u,v) con los parámetros del capítulo
- [ ] Filtrar celdas fuera del wound boundary
- [ ] Calcular centroides de celdas

#### 4.3 TSP/MILP Optimization
- [ ] Formular MILP con MTZ constraints (Eq. del capítulo)
- [ ] Implementar dummy depot node (open-path → closed-tour trick)
- [ ] Cost matrix: Euclidean + rise penalty (h_rise=20mm)
- [ ] Resolver con PuLP / scipy.milp (timeout 60s, fallback column-by-column)
- [ ] **DATOS A EXTRAER:**
  - Travel distance naive vs optimizado
  - % reducción
  - Mean solve time
  - Optimality gap
  - Fallback count

#### 4.4 Per-Cell Toolpath + 3D Mapping
- [ ] Implementar 5-phase sequence: approach → descend → perimeter trace (L layers) → center deposit → retract
- [ ] Mapear (u,v,h) → 3D via inverse conformal map
- [ ] Calcular normales y orientación de tool
- [ ] Two-stage deposition support (scaffold walls + hydrogel fill)
- [ ] **DATOS A EXTRAER:**
  - Wound coverage %
  - Travel-to-deposition ratio
  - Mean path curvature
  - Total waypoints per wound
  - Visualizaciones 3D del toolpath completo

---

### FASE 5: Módulo 4 — Motion Planning y Control (2 semanas)

**Objetivo:** IK + manipulability + PID para el sistema 8-DOF.

**Prioridad:** ALTA — cierra el loop de ejecución.

- [ ] Modelo cinemático 8-DOF con roboticstoolbox-python o implementación propia (numpy): 2 prismatic + 6 revolute UR5
- [ ] IK numérico con scipy.optimize (Levenberg-Marquardt) o roboticstoolbox-python
- [ ] Optimización de manipulability: explotar redundancia para maximizar μ(q)
- [ ] Collision avoidance geométrico (verificar distancias mínimas)
- [ ] PID velocity controller por joint (clase Python pura con numpy)
- [ ] Ejecutar trayectorias de las 20 wounds en CoppeliaSim (Python ZeroMQ API)
- [ ] **DATOS A EXTRAER:**
  - IK success rate (% waypoints resueltos)
  - Manipulability: mean/min μ para 8-DOF vs 6-DOF
  - % waypoints con μ < 0.005
  - Tracking error: mean/RMS/max position, mean orientation
  - Mean convergence time per waypoint
  - Deposition uniformity (σ layer thickness)
  - Plot de commanded vs achieved trajectory
  - Plot de manipulability profile

---

### FASE 6: Módulo 5 — Ejecución Closed-Loop (1-2 semanas)

**Objetivo:** Monitoreo visual durante impresión y verificación post-deposición.

**Prioridad:** MEDIA-ALTA — completa el pipeline pero depende de M1-M4.

- [ ] Implementar captura periódica durante ejecución (cada ~5 celdas)
- [ ] Re-evaluar wound coverage con CNN-Transformer en cada captura
- [ ] Implementar post-deposition verification (imagen final → Module 1 → coverage restante)
- [ ] **DATOS A EXTRAER:**
  - Planned coverage vs measured coverage
  - Coverage gap
  - Secuencia visual de cobertura progresiva

---

### FASE 7: Validación End-to-End (1-2 semanas)

**Objetivo:** Correr el pipeline completo de principio a fin, sin intervención, en las 20 wounds.

**Prioridad:** ALTA — es la prueba final.

- [ ] Pipeline automatizado: imagen → M1 → M2 → M3 → M4 → M5 → métricas
- [ ] Correr en 20 wounds sintéticas
- [ ] **DATOS A EXTRAER (Tabla e2e_summary):**
  - Todos los valores módulo por módulo
  - End-to-end pipeline time
  - Post-deposition coverage
- [ ] Comparación directa PoC baseline vs Proposed (Table poc_comparison)

---

### FASE 8 (TENTATIVA): Validación con Phantoms Físicos

**Objetivo:** Tests IRL. Deseable pero NO bloqueante para la tesis.

**Prioridad:** BAJA — nice-to-have, no es vital.

- [ ] Imprimir 3-5 phantoms en FDM (PLA) con geometrías de wound variadas
- [ ] Ejecutar pipeline con cámara real
- [ ] Depositar con marker ink como proxy de bioink
- [ ] Medir Chamfer, coverage, tracking error en phantoms
- [ ] Si no se hace: ajustar texto de Ch.4-6 para reflejar que phantom validation es trabajo futuro

---

### FASE 9: Cierre del Documento (2-3 semanas, en paralelo con fases finales)

**Objetivo:** Reemplazar todos los placeholders con datos reales y generar figuras definitivas.

#### 9.1 Reemplazar valores numéricos
- [ ] Ch.4 Results: sustituir los ~104 `\textcolor{red}{...}` con valores experimentales reales
- [ ] Ch.5 Discussion: actualizar ~34 valores que referencian resultados
- [ ] Ch.6 Conclusions: actualizar ~11 valores
- [ ] Abstract (EN + ES): actualizar valores clave

#### 9.2 Generar figuras reales (23 PLACEHOLDERs)
- [ ] Fig. pipeline_overview — diagrama del pipeline completo
- [ ] Fig. hardware_arch — arquitectura hardware 8-DOF
- [ ] Fig. cnn_transformer_arch — diagrama de la arquitectura CNN-Transformer
- [ ] Fig. ablation_comparison — comparación cualitativa de 3 decoders
- [ ] Fig. unet_training — curvas de loss U-Net
- [ ] Fig. unet_validation — grid de segmentaciones cualitativas
- [ ] Fig. gcode_generation — pasos del pipeline G-code
- [ ] Fig. gcode_output — trayectoria G-code en frame del robot
- [ ] Fig. poc_sim_execution — screenshots CoppeliaSim del PoC
- [ ] Fig. m1_training — curvas de loss CNN-Transformer
- [ ] Fig. m1_qualitative — predicciones cualitativas polar decoder
- [ ] Fig. m2_reconstruction — reconstrucción 3D + heatmap error
- [ ] Fig. m2_multiview — diagrama multi-view acquisition
- [ ] Fig. m3_honeycomb — lattice en (u,v) y en 3D
- [ ] Fig. m3_toolpath — toolpath 3D completo
- [ ] Fig. honeycomb_lattice — lattice generation
- [ ] Fig. tsp_optimization — naive vs optimizado
- [ ] Fig. per_cell_toolpath — secuencia 5-phase
- [ ] Fig. m4_manipulability — perfil de manipulability
- [ ] Fig. m4_tracking — commanded vs achieved
- [ ] Fig. m5_monitoring — monitoreo progresivo
- [ ] Fig. feedback_loop — diagrama closed-loop
- [ ] Fig. phantom_results — (solo si se hace Fase 8)

#### 9.3 Limpieza final
- [ ] Corregir entradas .bib con autores genéricos ("Researcher Names", "Author Names Unknown", "Various Authors") — hay ~6 entradas así
- [ ] Resolver las 5 notas "NOTE TO FUTURE ME":
  - Antecedents: verificar papers concurrentes 2025-2028
  - Results M2: confirmar approach elegido
  - Results M5: actualizar con datos de monitoreo
  - Results phantoms: actualizar o mover a future work
  - Discussion: verificar papers concurrentes
  - Conclusions: listar publicaciones reales
- [ ] Verificar citation-needed en Discussion (clinical procedure time)
- [ ] Compilar LaTeX completo y verificar formato
- [ ] Revisar consistencia de todos los cross-references entre capítulos

---

## Cronograma Sugerido

| Semana | Fase | Entregable |
|--------|------|------------|
| — | ~~F1: PoC Baseline~~ | **COMPLETADA (2025)** |
| 1 | F0: Setup | Entorno listo, datos descargados |
| 2-6 | F2: CNN-Transformer + Ablation | 3 decoders entrenados, ablation study completo |
| 7-8 | F3: Reconstrucción 3D | MVS pipeline en 20 wounds |
| 9-11 | F4: Trayectoria 3D | Honeycomb + TSP + toolpath completo |
| 12-13 | F5: Motion Planning | IK + manipulability + tracking results |
| 14-16 | F6-F7: Execution + E2E | Pipeline completo corriendo autónomamente |
| 16.5-18.5 | F8 (tentativa) | Phantoms si hay tiempo/recursos |
| 16-18.5 | F9: Cierre documento | Valores reales, figuras, compilación final |

**Total estimado: ~4.5 meses (~18 semanas)** (asumiendo dedicación de tiempo completo)

---

## Dependencias Críticas

```
F0 ──→ F1 (PoC) ──→ comparación final (F7)
              │
F0 ──→ F2 (CNN-Transformer) ──→ F3 (3D Recon) ──→ F4 (Trajectory) ──→ F5 (Robot) ──→ F6 (Execution) ──→ F7 (E2E)
                                                                                                            │
                                                                                                    F8 (tentativa)
                                                                                                            │
                                                                                                    F9 (cierre doc)
```

F1 y F2 pueden paralelizarse parcialmente (F1 primero, F2 empieza en semana 3-4).
F9 empieza en paralelo con F6-F7 (ir generando figuras conforme salen datos).

---

## Regla de Oro

> **Si los phantoms no se hacen, la tesis sigue siendo sólida.**
> El core es Ciencias Computacionales: la arquitectura, el ablation study, la generación de trayectoria conformal, y la validación in-silico.
> Los phantoms son la cereza, no el pastel.
> En ese caso, mover la sección de phantoms a "Future Work" y ajustar la narrativa de Ch.4-6 para reflejar que la validación fue exclusivamente in-silico con 20 modelos sintéticos de geometría conocida.
