Я написал статью по Flow Matching трансляции КТ изображений из одной серии (контрастной в неконтрастную).

Мне нужно сформулировать и доказать теорему по Flow Matching о возможности двунаправленной трансляции через реверс по времени.

1. Сформулируй теорему.
2. Докажи её.

---

## Ответ

### Теорема о двунаправленной трансляции в Flow Matching (Bidirectional Flow Matching Translation Theorem)

**Теорема.** Пусть $\pi_0$ и $\pi_1$ — распределения двух связанных модальностей изображений (например, native и contrast CT), и пусть $(\mathbf{x}_0, \mathbf{x}_1) \sim \pi_0 \times \pi_1$ — пара связанных изображений. Определим линейный интерполяционный путь:

$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1, \quad t \in [0,1]$$

и соответствующее целевое поле скоростей $\mathbf{v}_t = \mathbf{x}_1 - \mathbf{x}_0$.

Пусть $f_\theta(\mathbf{x}_t, t)$ — нейронная сеть, обученная аппроксимировать $\mathbf{v}_t$ минимизацией функции потерь Flow Matching:

$$\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1]} \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1) \sim \pi_0 \times \pi_1} \left[ \|f_\theta(\mathbf{x}_t, t) - \mathbf{v}_t\|_2^2 \right]$$

Тогда одна и та же обученная сеть $f_\theta$ позволяет выполнять **двунаправленную трансляцию**:

1. **Прямая трансляция** ($\pi_0 \to \pi_1$): Интегрирование ОДУ $\frac{d\mathbf{x}_t}{dt} = f_\theta(\mathbf{x}_t, t)$ от $t=0$ до $t=1$ с начальным условием $\mathbf{x}_0 \sim \pi_0$ даёт $\hat{\mathbf{x}}_1 \approx \mathbf{x}_1$.

2. **Обратная трансляция** ($\pi_1 \to \pi_0$): Интегрирование того же ОДУ от $t=1$ до $t=0$ с начальным условием $\mathbf{x}_1 \sim \pi_1$ даёт $\hat{\mathbf{x}}_0 \approx \mathbf{x}_0$.

---

### Доказательство

**Шаг 1: Анализ поля скоростей.**

Для линейного интерполяционного пути $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ истинное поле скоростей определяется как:

$$\mathbf{v}_t = \frac{\partial \mathbf{x}_t}{\partial t} = \frac{\partial}{\partial t}\left[(1-t)\mathbf{x}_0 + t\mathbf{x}_1\right] = -\mathbf{x}_0 + \mathbf{x}_1 = \mathbf{x}_1 - \mathbf{x}_0$$

Заметим, что $\mathbf{v}_t$ **не зависит от $t$** и является константой вдоль всей траектории для фиксированной пары $(\mathbf{x}_0, \mathbf{x}_1)$.

**Шаг 2: Прямая трансляция (Native → Contrast).**

Рассмотрим ОДУ с начальным условием $\mathbf{x}(0) = \mathbf{x}_0$:

$$\frac{d\mathbf{x}(t)}{dt} = f_\theta(\mathbf{x}(t), t) \approx \mathbf{v}_t = \mathbf{x}_1 - \mathbf{x}_0$$

При условии, что $f_\theta \approx \mathbf{v}_t$, интегрирование от $t=0$ до $t=1$ даёт:

$$\mathbf{x}(1) = \mathbf{x}(0) + \int_0^1 f_\theta(\mathbf{x}(t), t) \, dt \approx \mathbf{x}_0 + \int_0^1 (\mathbf{x}_1 - \mathbf{x}_0) \, dt = \mathbf{x}_0 + (\mathbf{x}_1 - \mathbf{x}_0) = \mathbf{x}_1$$

Таким образом, интегрирование в прямом направлении восстанавливает $\mathbf{x}_1$ из $\mathbf{x}_0$.

**Шаг 3: Обратная трансляция (Contrast → Native).**

Теперь рассмотрим обратное интегрирование с начальным условием $\mathbf{x}(1) = \mathbf{x}_1$.

Выполним замену переменной $s = 1 - t$, тогда $t = 1 - s$ и $dt = -ds$. При $t: 1 \to 0$ имеем $s: 0 \to 1$.

Обозначим $\mathbf{y}(s) = \mathbf{x}(1-s)$. Тогда:

$$\frac{d\mathbf{y}(s)}{ds} = \frac{d\mathbf{x}(1-s)}{ds} = -\frac{d\mathbf{x}(t)}{dt}\bigg|_{t=1-s} = -f_\theta(\mathbf{x}(1-s), 1-s) = -f_\theta(\mathbf{y}(s), 1-s)$$

Поскольку $f_\theta(\mathbf{y}(s), 1-s) \approx \mathbf{x}_1 - \mathbf{x}_0$, получаем:

$$\frac{d\mathbf{y}(s)}{ds} \approx -(\mathbf{x}_1 - \mathbf{x}_0) = \mathbf{x}_0 - \mathbf{x}_1$$

Интегрируя от $s=0$ до $s=1$ с начальным условием $\mathbf{y}(0) = \mathbf{x}(1) = \mathbf{x}_1$:

$$\mathbf{y}(1) = \mathbf{y}(0) + \int_0^1 (\mathbf{x}_0 - \mathbf{x}_1) \, ds = \mathbf{x}_1 + (\mathbf{x}_0 - \mathbf{x}_1) = \mathbf{x}_0$$

Поскольку $\mathbf{y}(1) = \mathbf{x}(0)$, мы восстанавливаем $\mathbf{x}_0$ из $\mathbf{x}_1$.

**Шаг 4: Эквивалентность прямого и обратного интегрирования.**

Ключевое наблюдение: обратное интегрирование ОДУ

$$\frac{d\mathbf{x}}{dt} = f_\theta(\mathbf{x}, t)$$

от $t=1$ до $t=0$ эквивалентно прямому интегрированию **того же** уравнения, но с обращённым направлением времени. Благодаря тому, что поле скоростей $\mathbf{v}_t = \mathbf{x}_1 - \mathbf{x}_0$ является константой (не зависит явно от $t$), нейронная сеть $f_\theta$, обученная предсказывать это поле, содержит всю необходимую информацию для трансляции в обоих направлениях.

**Заключение.** Одна нейронная сеть $f_\theta$, обученная на парах изображений $(\mathbf{x}_0, \mathbf{x}_1)$, способна выполнять двунаправленную трансляцию: интегрирование от $t=0$ к $t=1$ транслирует из $\pi_0$ в $\pi_1$, а интегрирование от $t=1$ к $t=0$ транслирует из $\pi_1$ в $\pi_0$. $\blacksquare$

---

### Замечания

1. **Практическое следствие:** Данная теорема объясняет, почему в статье одна модель (TimeResNet, SwinUNETR и др.) может выполнять трансляцию как из Contrast в Native, так и из Native в Contrast, что сокращает вычислительные затраты вдвое.

2. **Численное интегрирование:** На практике используются численные методы (Euler, RK2, RK4), которые вносят погрешность. Как показано в таблице Solver Ablation, выбор метода и количества шагов влияет на качество результата.

3. **Условие применимости:** Теорема справедлива при условии, что $f_\theta$ хорошо аппроксимирует истинное поле скоростей на всём пути $\mathbf{x}_t$ для $t \in [0,1]$.

---

### Лемма об ошибке аппроксимации (Approximation Error Bound Lemma)

**Лемма.** Пусть $\mathbf{v}(\mathbf{x}, t) = \mathbf{x}_1 - \mathbf{x}_0$ — истинное поле скоростей, а $f_\theta(\mathbf{x}, t)$ — его аппроксимация нейронной сетью. Пусть выполнены следующие условия:

1. **Равномерная ошибка аппроксимации:** Существует $\delta > 0$ такое, что для всех $\mathbf{x}$ в области определения и $t \in [0, 1]$:
   $$\|f_\theta(\mathbf{x}, t) - \mathbf{v}(\mathbf{x}, t)\|_2 \leq \delta$$

2. **Условие Липшица:** Функция $f_\theta$ является $L$-липшицевой по первому аргументу, т.е. для всех $\mathbf{x}, \mathbf{y}$ и $t \in [0, 1]$:
   $$\|f_\theta(\mathbf{x}, t) - f_\theta(\mathbf{y}, t)\|_2 \leq L \|\mathbf{x} - \mathbf{y}\|_2$$

Тогда ошибка трансляции $\|\hat{\mathbf{x}}_1 - \mathbf{x}_1\|_2 \leq \varepsilon$ гарантируется при условии:

$$\boxed{\delta \leq \frac{\varepsilon \cdot L}{e^L - 1}}$$

или, эквивалентно, ошибка трансляции ограничена сверху:

$$\boxed{\|\hat{\mathbf{x}}_1 - \mathbf{x}_1\|_2 \leq \frac{\delta}{L}(e^L - 1)}$$

**Частный случай (малая константа Липшица, $L \ll 1$):**

При $L \to 0$ имеем $\frac{e^L - 1}{L} \to 1$, и оценка упрощается до:

$$\|\hat{\mathbf{x}}_1 - \mathbf{x}_1\|_2 \leq \delta$$

То есть ошибка трансляции не превышает ошибку аппроксимации поля скоростей.

---

### Доказательство леммы

**Шаг 1: Постановка задачи.**

Пусть $\mathbf{x}(t)$ — истинная траектория, удовлетворяющая:
$$\frac{d\mathbf{x}(t)}{dt} = \mathbf{v}(\mathbf{x}(t), t), \quad \mathbf{x}(0) = \mathbf{x}_0$$

Пусть $\hat{\mathbf{x}}(t)$ — приближённая траектория, получаемая интегрированием аппроксимирующего поля:
$$\frac{d\hat{\mathbf{x}}(t)}{dt} = f_\theta(\hat{\mathbf{x}}(t), t), \quad \hat{\mathbf{x}}(0) = \mathbf{x}_0$$

Обе траектории начинаются из одной точки $\mathbf{x}_0$.

**Шаг 2: Уравнение для ошибки.**

Обозначим ошибку $\mathbf{e}(t) = \hat{\mathbf{x}}(t) - \mathbf{x}(t)$. Тогда:

$$\frac{d\mathbf{e}(t)}{dt} = f_\theta(\hat{\mathbf{x}}(t), t) - \mathbf{v}(\mathbf{x}(t), t)$$

Добавим и вычтем $f_\theta(\mathbf{x}(t), t)$:

$$\frac{d\mathbf{e}(t)}{dt} = \underbrace{\left[f_\theta(\hat{\mathbf{x}}(t), t) - f_\theta(\mathbf{x}(t), t)\right]}_{\text{(I)}} + \underbrace{\left[f_\theta(\mathbf{x}(t), t) - \mathbf{v}(\mathbf{x}(t), t)\right]}_{\text{(II)}}$$

**Шаг 3: Оценка слагаемых.**

Для слагаемого (I) используем условие Липшица:
$$\|\text{(I)}\|_2 = \|f_\theta(\hat{\mathbf{x}}(t), t) - f_\theta(\mathbf{x}(t), t)\|_2 \leq L \|\hat{\mathbf{x}}(t) - \mathbf{x}(t)\|_2 = L \|\mathbf{e}(t)\|_2$$

Для слагаемого (II) используем условие равномерной ошибки:
$$\|\text{(II)}\|_2 = \|f_\theta(\mathbf{x}(t), t) - \mathbf{v}(\mathbf{x}(t), t)\|_2 \leq \delta$$

**Шаг 4: Дифференциальное неравенство.**

Обозначим скалярную функцию $\eta(t) = \|\mathbf{e}(t)\|_2$ — норму вектора ошибки. Вычислим производную $\eta(t)$ (предполагая $\eta(t) > 0$):

$$\frac{d\eta(t)}{dt} = \frac{d}{dt}\|\mathbf{e}(t)\|_2 = \frac{d}{dt}\sqrt{\langle \mathbf{e}(t), \mathbf{e}(t) \rangle} = \frac{\langle \mathbf{e}(t), \frac{d\mathbf{e}(t)}{dt} \rangle}{\|\mathbf{e}(t)\|_2}$$

Применяя неравенство Коши-Шварца $|\langle \mathbf{a}, \mathbf{b} \rangle| \leq \|\mathbf{a}\|_2 \|\mathbf{b}\|_2$:

$$\frac{d\eta(t)}{dt} = \frac{\langle \mathbf{e}(t), \frac{d\mathbf{e}(t)}{dt} \rangle}{\|\mathbf{e}(t)\|_2} \leq \frac{\|\mathbf{e}(t)\|_2 \cdot \|\frac{d\mathbf{e}(t)}{dt}\|_2}{\|\mathbf{e}(t)\|_2} = \left\|\frac{d\mathbf{e}(t)}{dt}\right\|_2$$

Используя неравенство треугольника для нормы суммы (I) + (II) из Шага 3:

$$\left\|\frac{d\mathbf{e}(t)}{dt}\right\|_2 \leq \|\text{(I)}\|_2 + \|\text{(II)}\|_2 \leq L \cdot \eta(t) + \delta$$

Таким образом:

$$\frac{d\eta(t)}{dt} \leq L \cdot \eta(t) + \delta$$

с начальным условием $\eta(0) = 0$ (траектории стартуют из одной точки).

**Замечание:** При $\eta(t) = 0$ полагаем $\frac{d\eta}{dt} = \lim_{h \to 0^+} \frac{\eta(t+h) - \eta(t)}{h} \geq 0$, и неравенство остаётся справедливым, поскольку $\eta(t) \geq 0$ и $\eta(0) = 0$.

**Шаг 5: Применение леммы Гронуолла.**

Дифференциальное неравенство $\frac{d\eta}{dt} \leq L \cdot \eta + \delta$ с $\eta(0) = 0$ решается по лемме Гронуолла-Беллмана.

Общее решение линейного неоднородного ОДУ $\frac{d\eta}{dt} = L \cdot \eta + \delta$:

$$\eta(t) = e^{Lt}\left(\eta(0) + \int_0^t \delta \cdot e^{-Ls} ds\right) = e^{Lt} \cdot \frac{\delta}{L}\left(1 - e^{-Lt}\right) = \frac{\delta}{L}\left(e^{Lt} - 1\right)$$

**Шаг 6: Оценка ошибки в конечный момент времени.**

При $t = 1$ получаем:

$$\eta(1) = \|\hat{\mathbf{x}}(1) - \mathbf{x}(1)\|_2 \leq \frac{\delta}{L}(e^L - 1)$$

**Шаг 7: Условие на $\delta$ для достижения точности $\varepsilon$.**

Чтобы гарантировать $\|\hat{\mathbf{x}}_1 - \mathbf{x}_1\|_2 \leq \varepsilon$, требуется:

$$\frac{\delta}{L}(e^L - 1) \leq \varepsilon \quad \Longleftrightarrow \quad \delta \leq \frac{\varepsilon \cdot L}{e^L - 1}$$

$\blacksquare$

---

### Следствия леммы

**Следствие 1 (Таблица требуемой точности).**

| Константа Липшица $L$ | Множитель $\frac{e^L - 1}{L}$ | Требуемая $\delta$ для $\varepsilon = 1$ |
|:---------------------:|:-----------------------------:|:----------------------------------------:|
| 0.1                   | 1.052                         | 0.951                                    |
| 0.5                   | 1.297                         | 0.771                                    |
| 1.0                   | 1.718                         | 0.582                                    |
| 2.0                   | 3.195                         | 0.313                                    |
| 5.0                   | 29.48                         | 0.034                                    |

**Следствие 2 (Связь с функцией потерь).**

Функция потерь Flow Matching $\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}\left[\|f_\theta - \mathbf{v}\|_2^2\right]$ напрямую минимизирует $\delta^2$ в среднем. Если $\mathcal{L}_{\mathrm{FM}}(\theta) \leq \delta^2$, то $\delta = \sqrt{\mathcal{L}_{\mathrm{FM}}}$ может быть использовано для оценки ошибки трансляции.

**Следствие 3 (Практическая рекомендация).**

Для стабильной работы Flow Matching модели необходимо:
1. Минимизировать $\mathcal{L}_{\mathrm{FM}}$ (уменьшает $\delta$)
2. Использовать архитектуры с умеренной константой Липшица $L$ (BatchNorm, GroupNorm, спектральная нормализация)
3. При большом $L$ требуется экспоненциально более точная аппроксимация