import js from '@eslint/js';
  import pluginVue from 'eslint-plugin-vue';
  import globals from 'globals';

  export default [
    // Base ESLint recommended config
    js.configs.recommended,

    // Vue 3 recommended configs
    ...pluginVue.configs['flat/recommended'],

    {
      languageOptions: {
        ecmaVersion: 2021,
        sourceType: 'module',
        globals: {
          ...globals.browser,
          ...globals.node,
          ...globals.es2021,
        },
      },

      rules: {
        // Complexity rules
        'complexity': ['warn', 10],
        'max-depth': ['warn', 4],
        'max-lines-per-function': ['warn', 50],
      },
    },

    {
      files: ['**/*.vue'],
    },
  ];
