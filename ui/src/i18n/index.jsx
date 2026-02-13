import React, { createContext, useContext, useState, useCallback } from 'react';
import en from './locales/en.json';
import it from './locales/it.json';
import fr from './locales/fr.json';
import es from './locales/es.json';

const locales = { en, it, fr, es };

const LanguageContext = createContext();

export function LanguageProvider({ children, defaultLanguage = 'en' }) {
  const [language, setLanguage] = useState(() => {
    try {
      const saved = localStorage.getItem('antonio_language');
      return saved && locales[saved] ? saved : defaultLanguage;
    } catch { return defaultLanguage; }
  });

  const changeLanguage = useCallback((lang) => {
    if (locales[lang]) {
      setLanguage(lang);
      localStorage.setItem('antonio_language', lang);
    }
  }, []);

  return (
    <LanguageContext.Provider value={{ language, changeLanguage, languages: Object.keys(locales) }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useTranslation() {
  const { language, changeLanguage, languages } = useContext(LanguageContext);

  const t = useCallback((key, params = {}) => {
    const keys = key.split('.');
    let value = locales[language];
    for (const k of keys) {
      value = value?.[k];
    }
    if (typeof value !== 'string') return key;
    // Replace {{param}} placeholders
    return value.replace(/\{\{(\w+)\}\}/g, (_, name) => params[name] ?? `{{${name}}}`);
  }, [language]);

  return { t, language, changeLanguage, languages };
}
