# BPMN-диаграмма процесса автоматического переобучения модели

## Описание процесса
Данная диаграмма описывает автоматизированный цикл переобучения (Retraining) и замены модели классификации рака груди в продакшене, включая триггеры запуска, этапы обучения и критерии остановки.

## Диаграмма процесса (Mermaid)

```mermaid
flowchart TD
    %% Блок 1: Триггеры (Triggers)
    subgraph Triggers [1. Триггеры запуска переобучения]
        T1((Дрейф данных<br/>PSI > 0.2))
        T2((Падение метрик<br/>Accuracy < 95%))
        T3((Расписание<br/>Ежедневно в 02:00))
        T4((Новые данные<br/>> 1000 образцов))
        T5((Ручной запуск<br/>Data Scientist))
    end

    T1 & T2 & T3 & T4 & T5 --> Start([Начало процесса переобучения])

    %% Блок 2: Процесс переобучения (Retraining Process)
    subgraph Process [2. Процесс переобучения и валидации]
        Start --> Collect[Сбор новых и исторических данных]
        Collect --> DataCheck{Данных достаточно<br/>и они валидны?}
        DataCheck -->|Нет| StopData([Остановка: Ошибка качества данных])
        
        DataCheck -->|Да| Preprocess[Предобработка: IQR, StandardScaler, PCA]
        Preprocess --> InitCounter[Сброс счетчика попыток: Attempt = 1]
        InitCounter --> Train[Обучение моделей: Perceptron, KNN, DT, NB]
        Train --> Evaluate[Оценка на тестовой выборке]
        
        Evaluate --> Compare{Метрики лучше текущей<br/>модели в продакшне?<br/>Accuracy > 95%}
    end

    %% Блок 3: Критерии остановки и принятие решения (Stopping Criteria)
    subgraph Stopping [3. Критерии остановки и деплой]
        Compare -->|Да| Register[Регистрация новой версии в Model Registry]
        Register --> Shadow[Shadow / A/B тестирование]
        Shadow --> ABCheck{Успешно на реальном<br/>трафике?}
        
        ABCheck -->|Да| Deploy[Замена модели в продакшне]
        Deploy --> StopSuccess([Успешная остановка: Модель обновлена])
        
        ABCheck -->|Нет| Rollback[Откат к предыдущей версии]
        Rollback --> AlertFail[Уведомление ML-инженера]
        AlertFail --> StopFail([Остановка: Откат изменений])

        Compare -->|Нет| CheckAttempts{Достигнут лимит<br/>попыток?<br/>Attempt >= 3}
        CheckAttempts -->|Нет| Tune[Автоподбор гиперпараметров<br/>Attempt = Attempt + 1]
        Tune --> Train
        
        CheckAttempts -->|Да| AlertDS[Алерт Data Scientist: Ручной анализ]
        AlertDS --> StopLimit([Остановка: Исчерпан лимит попыток])
    end

    %% Стилизация блоков
    style Triggers fill:#f0f8ff,stroke:#4682b4,stroke-width:2px
    style Process fill:#f5fffa,stroke:#2e8b57,stroke-width:2px
    style Stopping fill:#fffaf0,stroke:#d2691e,stroke-width:2px
```

## Ключевые компоненты диаграммы

### 1. Триггеры (Triggers)
* **Data Drift:** Индекс стабильности популяции (`PSI > 0.2`).
* **Degradation:** Снижение точности (`Accuracy < 95%`) или полноты (`Recall < 95%`).
* **Scheduled:** Плановый запуск по расписанию (ежедневно в 02:00).
* **Data Volume:** Накопление более 1000 новых размеченных образцов.
* **Manual:** Принудительный запуск специалистом.

### 2. Критерии остановки (Stopping Criteria)
Процесс автоматически завершается в одном из следующих случаев:
1. **Успешное обновление (Success):** Новая модель превзошла текущую по метрикам, прошла A/B-тест и выведена в продакшн.
2. **Исчерпан лимит попыток (Max Attempts Reached):** Выполнено 3 цикла переобучения с тюнингом гиперпараметров без улучшения качества (передача задачи человеку).
3. **Ошибка валидации данных (Data Quality Failure):** Обнаружены критические аномалии или нехватка данных до начала обучения.
4. **Провал A/B-теста (Rollback):** Деградация метрик на реальном трафике — автоматический откат к стабильной версии.
